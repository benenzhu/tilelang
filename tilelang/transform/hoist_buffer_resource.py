"""Hoist make_wave_buffer_resource & fix AMD async wait counts.

On gfx950, cp_async_gs_lds<16> calls make_wave_buffer_resource() which
emits 4x readfirstlane per invocation. When the same global buffer is
used in an unrolled copy loop, the buffer resource descriptor can be
computed once and reused across all iterations.

Also fixes async wait counts for AMD: NVIDIA uses commit groups, but
AMD tracks each buffer_load individually via vmcnt. So wait_group(N)
must become vmcnt(N * loads_per_group) instead of vmcnt(N).
"""

from tvm import tir
from tvm.tir import (
    AttrStmt,
    Call,
    Evaluate,
    Var,
    PrimFunc,
    stmt_functor,
)
from tvm.tir.transform import prim_func_pass

from tilelang.utils.target import target_is_gfx950

# Op handles
_op_ptx_cp_async_lds = tir.op.Op.get("tl.ptx_cp_async_lds")
_op_ptx_cp_async_lds_rsrc = tir.op.Op.get("tl.ptx_cp_async_lds_rsrc")
_op_tvm_access_ptr = tir.op.Op.get("tir.tvm_access_ptr")
_op_ptx_cp_async = tir.op.Op.get("tir.ptx_cp_async")


def _extract_buffer_var(access_ptr_expr):
    """Extract the buffer data Var from a tvm_access_ptr call."""
    if not isinstance(access_ptr_expr, Call):
        return None
    if access_ptr_expr.op != _op_tvm_access_ptr:
        return None
    if len(access_ptr_expr.args) < 2:
        return None
    data_arg = access_ptr_expr.args[1]
    if isinstance(data_arg, Var):
        return data_arg
    return None


def _is_async_load_call(stmt):
    """Check if a statement is an async load call (any variant)."""
    if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
        return False
    op = stmt.value.op
    return op == _op_ptx_cp_async_lds or op == _op_ptx_cp_async_lds_rsrc or op == _op_ptx_cp_async


# ---------------------------------------------------------------------------
# Buffer resource hoisting
# ---------------------------------------------------------------------------

def _collect_buffer_vars(body):
    """Collect unique global buffer Vars from ptx_cp_async_lds calls."""
    buffer_vars = {}

    def _visitor(stmt):
        if isinstance(stmt, Evaluate) and isinstance(stmt.value, Call):
            if stmt.value.op == _op_ptx_cp_async_lds:
                buf_var = _extract_buffer_var(stmt.value.args[1])
                if buf_var is not None and buf_var not in buffer_vars:
                    rsrc_var = Var("__rsrc_" + buf_var.name, dtype="handle")
                    base_var = Var("__base_" + buf_var.name, dtype="uint32")
                    buffer_vars[buf_var] = (rsrc_var, base_var)

    stmt_functor.post_order_visit(body, _visitor)
    return buffer_vars


def _rewrite_calls(body, buffer_vars):
    """Rewrite ptx_cp_async_lds -> ptx_cp_async_lds_rsrc with rsrc + base."""

    def _postorder(op):
        if isinstance(op, Evaluate) and isinstance(op.value, Call):
            if op.value.op == _op_ptx_cp_async_lds:
                buf_var = _extract_buffer_var(op.value.args[1])
                if buf_var is not None and buf_var in buffer_vars:
                    rsrc_var, base_var = buffer_vars[buf_var]
                    new_call = Call(
                        op.value.dtype,
                        _op_ptx_cp_async_lds_rsrc,
                        [op.value.args[0], op.value.args[1], op.value.args[2],
                         rsrc_var, base_var],
                    )
                    return Evaluate(new_call)
        return None

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.Evaluate"])


# ---------------------------------------------------------------------------
# AMD async wait count fix
# ---------------------------------------------------------------------------

def _count_async_loads(stmt, multiplier=1):
    """Count async load calls in a subtree, multiplying by loop extents."""
    if _is_async_load_call(stmt):
        return multiplier

    if isinstance(stmt, tir.For):
        ext = multiplier
        if isinstance(stmt.extent, tir.IntImm):
            ext = multiplier * stmt.extent.value
        return _count_async_loads(stmt.body, ext)

    if isinstance(stmt, tir.SeqStmt):
        return sum(_count_async_loads(s, multiplier) for s in stmt.seq)

    if isinstance(stmt, tir.AttrStmt):
        return _count_async_loads(stmt.body, multiplier)

    if isinstance(stmt, tir.IfThenElse):
        # Take the max of both branches
        count = _count_async_loads(stmt.then_case, multiplier)
        if stmt.else_case is not None:
            count = max(count, _count_async_loads(stmt.else_case, multiplier))
        return count

    if isinstance(stmt, tir.LetStmt):
        return _count_async_loads(stmt.body, multiplier)

    return 0


def _contains_commit_scope(stmt):
    """Check if a subtree contains an async_commit_queue_scope."""
    found = [False]

    def _v(s):
        if isinstance(s, tir.AttrStmt) and s.attr_key == "async_commit_queue_scope":
            found[0] = True

    stmt_functor.post_order_visit(stmt, _v)
    return found[0]


def _find_for_with_commit(stmt):
    """Find the innermost For loop whose body contains a commit scope."""
    if isinstance(stmt, tir.For):
        # Check if a deeper For also has a commit
        inner = _find_for_with_commit(stmt.body)
        if inner is not None:
            return inner
        if _contains_commit_scope(stmt.body):
            return stmt
    elif isinstance(stmt, tir.SeqStmt):
        for s in stmt.seq:
            result = _find_for_with_commit(s)
            if result is not None:
                return result
    elif hasattr(stmt, 'body'):
        # Handles AttrStmt, LetStmt, DeclBuffer, Allocate, etc.
        return _find_for_with_commit(stmt.body)
    return None


def _get_loads_per_group(body):
    """Count async loads per commit group.

    NVIDIA commit groups everything since the last commit. The commit
    scope in TIR only wraps the LAST batch of loads, but earlier loads
    (e.g. A copies) are outside the scope yet still in the same group.

    So we find the For loop containing the commit and count ALL async
    loads in one iteration of that loop.
    """
    for_node = _find_for_with_commit(body)
    if for_node is not None:
        return _count_async_loads(for_node.body)
    return 0


def _fix_amd_wait_counts(body, loads_per_group):
    """Replace async_wait_inflight_count N with N * loads_per_group.

    AMD has no commit groups — each buffer_load is tracked individually
    by vmcnt. So "keep N groups in flight" = vmcnt(N * loads_per_group).
    N=0 (wait all) stays vmcnt(0), which is correct.
    """

    def _postorder(op):
        if isinstance(op, tir.AttrStmt):
            if op.attr_key == "async_wait_inflight_count":
                if isinstance(op.value, tir.IntImm):
                    old_n = op.value.value
                    if old_n > 0:
                        new_n = old_n * loads_per_group
                        return tir.AttrStmt(
                            op.node, op.attr_key,
                            tir.IntImm("int32", new_n), op.body)
        return None

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.AttrStmt"])


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def HoistBufferResource():
    """Hoist buffer resource descriptors & fix async wait counts (ROCm gfx950)."""

    def pass_fn(func: PrimFunc, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func

        # --- Fix AMD async wait counts (before wrapping in AttrStmts) ---
        loads_per_group = _get_loads_per_group(func.body)

        # --- Buffer resource hoisting ---
        buffer_vars = _collect_buffer_vars(func.body)

        if buffer_vars:
            new_body = _rewrite_calls(func.body, buffer_vars)

            for buf_var, (rsrc_var, base_var) in reversed(list(buffer_vars.items())):
                new_body = AttrStmt(base_var, "buffer_base_var", buf_var, new_body)
                new_body = AttrStmt(rsrc_var, "buffer_resource_var", buf_var, new_body)
        else:
            new_body = func.body

        # --- Apply wait count fix ---
        if loads_per_group > 1:
            new_body = _fix_amd_wait_counts(new_body, loads_per_group)

        if new_body is not func.body:
            return func.with_body(new_body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)

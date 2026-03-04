"""Hoist make_wave_buffer_resource & fix AMD async wait counts.

On gfx950, cp_async_gs_lds<16> calls make_wave_buffer_resource() which
emits 4x readfirstlane per invocation. When the same global buffer is
used in an unrolled copy loop, the buffer resource descriptor can be
computed once and reused across all iterations.

Also fixes async wait counts for AMD: NVIDIA uses commit groups, but
AMD tracks each buffer_load individually via vmcnt. So wait_group(N)
must become vmcnt(N * loads_per_group) instead of vmcnt(N).

Additionally, hoists XOR-swizzle delta offsets from global addresses in
async copy calls. The swizzle delta only depends on threadIdx.x and is
loop-invariant, so computing it once avoids redundant VALU instructions
in the main k-loop.
"""

import os
import sys
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
# Swizzle delta hoisting
# ---------------------------------------------------------------------------

def _collect_vars(expr):
    """Collect all Var nodes referenced in an expression."""
    var_set = set()
    def _visitor(e):
        if isinstance(e, tir.Var):
            var_set.add(e)
    stmt_functor.post_order_visit(expr, _visitor)
    return var_set


def _extract_access_ptr_offset(access_ptr_expr):
    """Extract the byte offset from a tvm_access_ptr call.

    tvm_access_ptr(dtype, data, offset, extent, rw) -> returns offset.
    """
    if not isinstance(access_ptr_expr, Call):
        return None
    if access_ptr_expr.op != _op_tvm_access_ptr:
        return None
    if len(access_ptr_expr.args) < 3:
        return None
    return access_ptr_expr.args[2]


def _extract_thread_invariant(expr, thread_var):
    """Extract the sum of all thread_binding-only terms from an Add chain.

    Splits an additive expression into two parts:
      - thread_sum: terms that ONLY depend on thread_var (loop-invariant)
      - remainder:  everything else (depends on k, blockIdx, i, etc.)

    Returns (thread_sum, remainder).  thread_sum is None when no
    thread-only terms are found.
    """
    thread_terms = []
    other_terms = []

    def _collect_additive_terms(e, terms):
        """Flatten an Add chain into individual terms."""
        if isinstance(e, tir.Add):
            _collect_additive_terms(e.a, terms)
            _collect_additive_terms(e.b, terms)
        else:
            terms.append(e)

    all_terms = []
    _collect_additive_terms(expr, all_terms)

    for term in all_terms:
        used = _collect_vars(term)
        # A term qualifies if it uses thread_var and nothing else
        if used and all(v.same_as(thread_var) for v in used):
            thread_terms.append(term)
        else:
            other_terms.append(term)

    if not thread_terms:
        return None, expr

    # Build the thread-only sum
    thread_sum = thread_terms[0]
    for t in thread_terms[1:]:
        thread_sum = tir.Add(thread_sum, t)

    # Build the remainder
    if other_terms:
        remainder = other_terms[0]
        for t in other_terms[1:]:
            remainder = tir.Add(remainder, t)
    else:
        remainder = tir.IntImm("int32", 0)

    return thread_sum, remainder


def _find_thread_binding_var(body):
    """Find the thread_binding variable (threadIdx.x) from the function body.

    Looks for the IterVar bound to 'threadIdx.x' in thread_extent attrs.
    Falls back to the variable named 'thread_binding' if found.
    """
    thread_var = [None]

    def _visitor(stmt):
        if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "thread_extent":
            if isinstance(stmt.node, tir.IterVar):
                tag = stmt.node.thread_tag
                if tag == "threadIdx.x":
                    thread_var[0] = stmt.node.var

    stmt_functor.post_order_visit(body, _visitor)

    # Fallback: look for a Var named "thread_binding" in the body
    if thread_var[0] is None:
        def _var_visitor(e):
            if isinstance(e, tir.Var) and e.name == "thread_binding":
                thread_var[0] = e
        stmt_functor.post_order_visit(body, _var_visitor)

    return thread_var[0]


def _collect_thread_offsets(body):
    """Collect thread-only offset expressions from ptx_cp_async_lds_rsrc calls.

    Returns a dict mapping the canonical expression string to
    (Var, PrimExpr) for each unique thread-only offset found.
    """
    thread_var = _find_thread_binding_var(body)
    if thread_var is None:
        return {}, body

    offsets = {}  # str(expr) -> (var, expr)

    def _visitor(stmt):
        if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
            return
        if stmt.value.op != _op_ptx_cp_async_lds_rsrc:
            return
        offset = _extract_access_ptr_offset(stmt.value.args[1])
        if offset is None:
            return
        thread_sum, _ = _extract_thread_invariant(offset, thread_var)
        if thread_sum is None:
            return
        key = str(thread_sum)
        if key not in offsets:
            delta_var = Var("__g2s_thread_offset", dtype="int32")
            offsets[key] = (delta_var, thread_sum)

    stmt_functor.post_order_visit(body, _visitor)
    return offsets, body




def _inject_thread_offset_attrs(body, offsets):
    """Insert swizzle_delta_var AttrStmts inside the innermost thread_extent scope.

    We must place them inside thread_extent so that 'thread_binding' is
    already defined when codegen emits the expression.
    """
    injected = [False]

    def _postorder(op):
        if injected[0]:
            return None
        if isinstance(op, tir.AttrStmt) and op.attr_key == "thread_extent":
            if isinstance(op.node, tir.IterVar):
                tag = op.node.thread_tag
                if tag == "threadIdx.x":
                    inner = op.body
                    for _, (delta_var, delta_expr) in offsets.items():
                        inner = AttrStmt(delta_var, "swizzle_delta_var",
                                         delta_expr, inner)
                    injected[0] = True
                    return AttrStmt(op.node, op.attr_key, op.value, inner)
        return None

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.AttrStmt"])


def _rewrite_thread_offsets(body, offsets, thread_var):
    """Replace thread-only offset expressions in async copy calls with hoisted vars."""
    if not offsets:
        return body
    str_to_var = {k: v[0] for k, v in offsets.items()}

    def _postorder(op):
        if not isinstance(op, Evaluate) or not isinstance(op.value, Call):
            return None
        if op.value.op != _op_ptx_cp_async_lds_rsrc:
            return None
        src_access = op.value.args[1]
        offset = _extract_access_ptr_offset(src_access)
        if offset is None:
            return None
        thread_sum, remainder = _extract_thread_invariant(offset, thread_var)
        if thread_sum is None:
            return None
        key = str(thread_sum)
        if key not in str_to_var:
            return None
        delta_var = str_to_var[key]
        new_offset = tir.Add(remainder, delta_var)
        new_src = Call(
            src_access.dtype,
            src_access.op,
            [src_access.args[0], src_access.args[1], new_offset,
             src_access.args[3], src_access.args[4]],
        )
        new_call = Call(
            op.value.dtype,
            op.value.op,
            [op.value.args[0], new_src, op.value.args[2],
             op.value.args[3], op.value.args[4]],
        )
        return Evaluate(new_call)

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.Evaluate"])


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def HoistBufferResource():
    """Hoist buffer resource descriptors & fix async wait counts (ROCm gfx950).
    Currenlty, only loop one time to find a for, maybe bug when script contains multiple for loops. 
    TODO(zty): may have bug with the latter loops? maybe move this to the 
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func
        mod = None
        new_body = func.body
        # step 1: hoist buffer resource descriptors
        buffer_vars = _collect_buffer_vars(func.body)

        if buffer_vars:
            new_body = _rewrite_calls(func.body, buffer_vars)

            for buf_var, (rsrc_var, base_var) in reversed(list(buffer_vars.items())):
                new_body = AttrStmt(base_var, "buffer_base_var", buf_var, new_body)
                new_body = AttrStmt(rsrc_var, "buffer_resource_var", buf_var, new_body)
        else:
            new_body = func.body
        # step 2: fix AMD async wait counts
        loads_per_group = _get_loads_per_group(new_body)
        if loads_per_group > 1:
            new_body = _fix_amd_wait_counts(new_body, loads_per_group)

        # step 3: hoist thread-only offsets from async copy global addresses
        # Extracts ALL additive terms that only depend on threadIdx.x
        # (swizzle XOR + row offset etc.) into a precomputed variable.
        # The AttrStmt must be placed INSIDE the thread_extent scope
        # (where thread_binding is defined) so codegen can resolve it.
        thread_var = _find_thread_binding_var(new_body)
        offsets, _ = _collect_thread_offsets(new_body)
        if offsets and thread_var is not None:
            new_body = _rewrite_thread_offsets(new_body, offsets, thread_var)
            new_body = _inject_thread_offset_attrs(new_body, offsets)

        if new_body is not func.body:
            return func.with_body(new_body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)

"""Hoist make_wave_buffer_resource out of cp_async_gs_lds calls.

On gfx950, cp_async_gs_lds<16> calls make_wave_buffer_resource() which
emits 4x readfirstlane per invocation. When the same global buffer is
used in an unrolled copy loop, the buffer resource descriptor can be
computed once and reused across all iterations.

This pass:
1. Collects all ptx_cp_async_lds calls and extracts their source buffer var
2. For each unique buffer, emits AttrStmts to declare rsrc + base variables
3. Rewrites ptx_cp_async_lds -> ptx_cp_async_lds_rsrc(dst, src, bytes, rsrc, base)
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


def _collect_buffer_vars(body):
    """Collect unique global buffer Vars from ptx_cp_async_lds calls.

    Returns dict: buf_var -> (rsrc_var, base_var)
    """
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


def HoistBufferResource():
    """Hoist buffer resource descriptors for async G2S LDS copies (ROCm gfx950)."""

    def pass_fn(func: PrimFunc, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func

        # Step 1: Collect
        buffer_vars = _collect_buffer_vars(func.body)
        if not buffer_vars:
            return func

        # Step 2: Rewrite calls
        new_body = _rewrite_calls(func.body, buffer_vars)

        # Step 3: Wrap in AttrStmts (rsrc + base per buffer)
        for buf_var, (rsrc_var, base_var) in reversed(list(buffer_vars.items())):
            new_body = AttrStmt(base_var, "buffer_base_var", buf_var, new_body)
            new_body = AttrStmt(rsrc_var, "buffer_resource_var", buf_var, new_body)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

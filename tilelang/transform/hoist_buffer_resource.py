"""Hoist make_wave_buffer_resource out of cp_async_gs_lds calls.

On gfx950, cp_async_gs_lds<16> calls make_wave_buffer_resource() which
emits 4x readfirstlane per invocation. When the same global buffer is
used in an unrolled copy loop, the buffer resource descriptor can be
computed once and reused across all iterations.

This pass:
1. Collects all ptx_cp_async_lds calls and extracts their source buffer var
2. Groups them by source buffer variable
3. For each unique buffer, creates a LetStmt binding a rsrc variable
   to ptx_make_buffer_resource(buffer_ptr)
4. Rewrites ptx_cp_async_lds -> ptx_cp_async_lds_rsrc with the rsrc var
"""

from tvm import tir
from tvm.tir import (
    Call,
    Evaluate,
    LetStmt,
    Var,
    PrimFunc,
    PyStmtExprMutator,
    PyStmtExprVisitor,
)
from tvm.tir.transform import prim_func_pass

from tilelang.utils.target import target_is_gfx950

# Op handles for the intrinsics we care about
_op_ptx_cp_async_lds = tir.op.Op.get("tl.ptx_cp_async_lds")
_op_ptx_cp_async_lds_rsrc = tir.op.Op.get("tl.ptx_cp_async_lds_rsrc")
_op_ptx_make_buffer_resource = tir.op.Op.get("tl.ptx_make_buffer_resource")
_op_tvm_access_ptr = tir.op.Op.get("tir.tvm_access_ptr")


def _extract_buffer_var(access_ptr_expr):
    """Extract the buffer data Var from a tvm_access_ptr call.

    tvm_access_ptr(dtype, data, offset, extent, rw_mask)
    Returns data (args[1]) as a Var, or None.
    """
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


@tir.functor.visitor
class _BufferResourceCollector(PyStmtExprVisitor):
    """Collect all unique global buffer Vars used in ptx_cp_async_lds calls."""

    def __init__(self):
        super().__init__()
        # Map: buffer Var -> rsrc Var (preserves insertion order in Python 3.7+)
        self.buffer_to_rsrc = {}

    def visit_evaluate_(self, op):
        if isinstance(op.value, Call) and op.value.op == _op_ptx_cp_async_lds:
            # args[1] = src_access_ptr
            buf_var = _extract_buffer_var(op.value.args[1])
            if buf_var is not None and buf_var not in self.buffer_to_rsrc:
                rsrc_name = "__rsrc_" + buf_var.name
                rsrc_var = Var(rsrc_name, dtype="handle")
                self.buffer_to_rsrc[buf_var] = rsrc_var
        # Continue visiting children
        super().visit_evaluate_(op)


@tir.functor.mutator
class _BufferResourceRewriter(PyStmtExprMutator):
    """Rewrite ptx_cp_async_lds -> ptx_cp_async_lds_rsrc using pre-collected rsrc vars."""

    def __init__(self, buffer_to_rsrc):
        super().__init__()
        self._buffer_to_rsrc = buffer_to_rsrc

    def visit_evaluate_(self, op):
        if isinstance(op.value, Call) and op.value.op == _op_ptx_cp_async_lds:
            buf_var = _extract_buffer_var(op.value.args[1])
            if buf_var is not None and buf_var in self._buffer_to_rsrc:
                rsrc_var = self._buffer_to_rsrc[buf_var]
                new_call = Call(
                    op.value.dtype,
                    _op_ptx_cp_async_lds_rsrc,
                    [op.value.args[0], op.value.args[1], op.value.args[2], rsrc_var],
                )
                return Evaluate(new_call)
        return super().visit_evaluate_(op)


def HoistBufferResource():
    """Hoist buffer resource descriptors for async G2S LDS copies (ROCm gfx950).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        # Only apply on gfx950 targets
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func

        # Step 1: Collect all ptx_cp_async_lds calls and their source buffers
        collector = _BufferResourceCollector()
        collector.visit_stmt(func.body)

        if not collector.buffer_to_rsrc:
            return func

        # Step 2: Rewrite calls
        rewriter = _BufferResourceRewriter(collector.buffer_to_rsrc)
        new_body = rewriter.visit_stmt(func.body)

        # Step 3: Wrap body in LetStmts for rsrc variables
        # Reverse so first buffer is outermost
        for buf_var, rsrc_var in reversed(list(collector.buffer_to_rsrc.items())):
            make_rsrc = Call("handle", _op_ptx_make_buffer_resource, [buf_var])
            new_body = LetStmt(rsrc_var, make_rsrc, new_body)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

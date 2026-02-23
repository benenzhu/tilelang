"""Remove redundant tvm_storage_sync calls from interleaved pipeline loops.

When InjectCdna4Pipeline inserts explicit vmcnt + s_barrier synchronization,
the ThreadSync pass may have already inserted tvm_storage_sync("shared.dyn")
calls that are redundant and harmful (they translate to __syncthreads which
does implicit vmcnt(0), breaking the carefully planned vmcnt schedule).

This pass removes tvm_storage_sync calls from loop bodies that contain
explicit vmcnt calls.
"""

from __future__ import annotations

from tvm import tir
from tvm.tir import (
    AttrStmt,
    Evaluate,
    For,
    ForKind,
    IntImm,
    PrimFunc,
    SeqStmt,
    Stmt,
)
from tvm.tir.transform import prim_func_pass


def _is_storage_sync(stmt: Stmt) -> bool:
    """Check if stmt is Evaluate(tvm_storage_sync(...))."""
    if not isinstance(stmt, Evaluate):
        return False
    call = stmt.value
    if not isinstance(call, tir.Call):
        return False
    if not hasattr(call.op, 'name'):
        return False
    return call.op.name == "tir.tvm_storage_sync"


def _has_vmcnt_in_loop(stmt: Stmt) -> bool:
    """Check if a For loop body contains a tl_waitcnt_vmcnt call."""
    if isinstance(stmt, Evaluate):
        call = stmt.value
        if isinstance(call, tir.Call) and hasattr(call.op, 'name'):
            if call.op.name == "tir.call_extern" and len(call.args) > 0:
                arg0 = call.args[0]
                if isinstance(arg0, tir.StringImm) and arg0.value == "tl_waitcnt_vmcnt":
                    return True
        return False
    if isinstance(stmt, SeqStmt):
        return any(_has_vmcnt_in_loop(s) for s in stmt)
    if isinstance(stmt, For):
        return _has_vmcnt_in_loop(stmt.body)
    if isinstance(stmt, AttrStmt):
        return _has_vmcnt_in_loop(stmt.body)
    return False


def _remove_syncs(stmt: Stmt) -> Stmt | None:
    """Remove tvm_storage_sync from a statement tree. Returns None to signal removal."""
    if _is_storage_sync(stmt):
        return None

    if isinstance(stmt, SeqStmt):
        new_stmts = []
        changed = False
        for s in stmt:
            result = _remove_syncs(s)
            if result is None:
                changed = True
            else:
                new_stmts.append(result)
                if not result.same_as(s):
                    changed = True
        if not changed:
            return stmt
        if len(new_stmts) == 0:
            return Evaluate(IntImm("int32", 0))  # noop
        if len(new_stmts) == 1:
            return new_stmts[0]
        return SeqStmt(new_stmts)

    if isinstance(stmt, For):
        new_body = _remove_syncs(stmt.body)
        if new_body is None:
            return stmt
        if new_body.same_as(stmt.body):
            return stmt
        return For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            new_body, stmt.thread_binding, stmt.annotations,
        )

    if isinstance(stmt, AttrStmt):
        new_body = _remove_syncs(stmt.body)
        if new_body is None:
            # If body is removed, remove the entire attr
            return None
        if new_body.same_as(stmt.body):
            return stmt
        return AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)

    return stmt


@tir.functor.mutator
class RemoveRedundantSyncMutator(tir.PyStmtExprMutator):
    """Remove tvm_storage_sync from For loops that have explicit vmcnt + s_barrier."""

    def visit_for_(self, op: For) -> Stmt:
        # Only process serial loops that contain vmcnt (our interleaved loops)
        if op.kind == ForKind.SERIAL and _has_vmcnt_in_loop(op):
            new_body = _remove_syncs(op.body)
            if new_body is not None and not new_body.same_as(op.body):
                return For(
                    op.loop_var, op.min, op.extent, op.kind,
                    new_body, op.thread_binding, op.annotations,
                )
            return op

        # Default: recurse
        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op
        return For(
            op.loop_var, op.min, op.extent, op.kind,
            new_body, op.thread_binding, op.annotations,
        )


def RemoveRedundantSync():
    """TVM Pass: Remove redundant tvm_storage_sync from interleaved pipeline loops."""

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = RemoveRedundantSyncMutator()
        new_body = mutator.visit_stmt(func.body)
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

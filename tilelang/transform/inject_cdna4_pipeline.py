"""CDNA4 (gfx950) optimized pipeline: interleaved G2S with direct vmcnt.

Runs AFTER InjectSoftwarePipeline + LowerOpaqueBlock + Simplify.
Transforms the standard pipeline (all G2S → wait → compute) into
a HipKittens-style interleaved schedule where G2S loads are distributed
across MFMA compute phases.

Uses direct `tl_waitcnt_vmcnt(N)` and `s_barrier` calls, completely
bypassing InjectPTXAsyncCopy's commit-group vmcnt mapping.

G2S stores are emitted as raw async_scope stores (no async_commit_queue_scope),
so InjectPTXAsyncCopy won't insert spurious __syncthreads between them.
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
from tvm.tir.stmt_functor import substitute
from tvm.tir.transform import prim_func_pass

# Reuse helpers from interleave_g2s
from .interleave_g2s import (
    _flatten_seq,
    _is_sched_barrier,
    _find_async_blocks,
    _find_consumer,
    _extract_consumer_stages,
    _rewrap,
)


def _make_vmcnt(n: int) -> Stmt:
    """Emit asm volatile s_waitcnt vmcnt(N) directly."""
    return Evaluate(
        tir.call_extern("int32", "tl_waitcnt_vmcnt", IntImm("int32", n))
    )


def _make_s_barrier() -> Stmt:
    """Emit __builtin_amdgcn_s_barrier() for workgroup sync."""
    return Evaluate(
        tir.call_extern("int32", "__builtin_amdgcn_s_barrier")
    )


def _make_raw_g2s(inner_store_loop: Stmt) -> Stmt:
    """Wrap a G2S store in async_scope only (NO async_commit_queue_scope)."""
    return AttrStmt(
        IntImm("int32", 0), "async_scope", IntImm("int32", 1),
        inner_store_loop,
    )


def _interleave_one_stage(
    stage_stmts: list[Stmt],
    g2s_stores: list[Stmt],
    vmcnt_value: int | None = None,
    barrier_after: bool = False,
) -> list[Stmt]:
    """Insert G2S loads into one MFMA stage, with optional vmcnt before MFMA."""
    sched_barrier_count = 0
    insert_idx = None

    for j, s in enumerate(stage_stmts):
        if _is_sched_barrier(s):
            sched_barrier_count += 1
            if sched_barrier_count == 2:
                insert_idx = j
                break

    if insert_idx is None:
        return stage_stmts

    result = []
    result.extend(stage_stmts[:insert_idx])
    for g in g2s_stores:
        result.append(_make_raw_g2s(g))
    result.append(stage_stmts[insert_idx])
    if vmcnt_value is not None:
        result.append(_make_vmcnt(vmcnt_value))
    if barrier_after:
        result.append(_make_s_barrier())
    result.extend(stage_stmts[insert_idx + 1:])

    return result


def _try_cdna4_pipeline(loop: For) -> Stmt | None:
    """Transform pipeline loop body into interleaved stages.

    Returns the new loop body (For node) with G2S distributed across
    MFMA stages, or None if the loop doesn't match.

    The caller (_transform_pipeline_seq) handles removing the original G2S
    and replacing the async_wait synchronization.
    """
    body_stmts = _flatten_seq(loop.body)
    k = loop.loop_var

    # 1. Extract G2S and consumer
    g2s_a, g2s_b, remaining = _find_async_blocks(body_stmts)
    if g2s_a is None or g2s_b is None:
        return None

    consumer_body, _found = _find_consumer(remaining)
    if consumer_body is None:
        return None

    result = _extract_consumer_stages(consumer_body)
    if result is None:
        return None
    mfma_stages, alloc_wrappers = result

    if len(mfma_stages) != 4:
        return None

    total_g2s = g2s_a + g2s_b
    num_g2s = len(total_g2s)
    if num_g2s == 0:
        return None

    orig_n = int(loop.extent) if isinstance(loop.extent, IntImm) else None
    if orig_n is None or orig_n < 2:
        return None

    # 2. Build interleaved stages with G2S distributed
    g2s_per_stage = (num_g2s + 3) // 4

    phase_stmts = []
    for i, stage in enumerate(mfma_stages):
        g_start = i * g2s_per_stage
        g_end = min(g_start + g2s_per_stage, num_g2s)
        g2s_slice = total_g2s[g_start:g_end]

        is_last = (i == len(mfma_stages) - 1)
        phase_stmts.extend(_interleave_one_stage(
            stage,
            g2s_slice,
            vmcnt_value=0 if is_last else None,
            barrier_after=is_last,
        ))

    interleaved = SeqStmt(phase_stmts)
    if alloc_wrappers:
        interleaved = _rewrap(interleaved, alloc_wrappers)

    # 3. New loop body: just the interleaved stages (no original G2S, no async_wait)
    return For(
        k, loop.min, loop.extent,
        loop.kind, interleaved, loop.thread_binding, loop.annotations,
    )


def _is_pipeline_loop(stmt: Stmt) -> bool:
    """Check if a For looks like a pipeline loop (serial, extent>1, has async)."""
    if not isinstance(stmt, For) or stmt.kind != ForKind.SERIAL:
        return False
    if not isinstance(stmt.extent, IntImm) or int(stmt.extent) <= 1:
        return False
    body_stmts = _flatten_seq(stmt.body)
    return any(
        isinstance(s, AttrStmt)
        and s.attr_key in ("async_scope", "async_commit_queue_scope")
        for s in body_stmts
    )


def _transform_pipeline_seq(stmts: list[Stmt]):
    """Transform a sequence containing [prologue_g2s..., k_loop, epilogue].

    Finds the pipeline k-loop, applies interleaving, and removes original G2S
    from the loop body while keeping prologue G2S + adding vmcnt(0)+s_barrier.

    Returns transformed list of statements, or None if no transformation was done.
    """
    # Find the pipeline k-loop index
    loop_idx = None
    for i, s in enumerate(stmts):
        if _is_pipeline_loop(s):
            loop_idx = i
            break

    if loop_idx is None:
        return None

    loop = stmts[loop_idx]
    new_loop = _try_cdna4_pipeline(loop)
    if new_loop is None:
        return None

    # Build the new statement sequence:
    # 1. Keep everything before the loop (prologue G2S, etc.)
    # 2. Add vmcnt(0) + s_barrier to wait for prologue G2S
    # 3. The transformed loop (with interleaved G2S, no original G2S)
    # 4. Keep everything after the loop (epilogue)
    result = []
    result.extend(stmts[:loop_idx])       # prologue G2S blocks
    result.append(_make_vmcnt(0))          # wait for prologue G2S
    result.append(_make_s_barrier())       # workgroup sync
    result.append(new_loop)                # interleaved loop
    result.extend(stmts[loop_idx + 1:])   # epilogue
    return result


@tir.functor.mutator
class InjectCdna4PipelineMutator(tir.PyStmtExprMutator):
    """Find the pipeline sequence and apply CDNA4 interleaved schedule.

    Works at the SeqStmt level to transform the entire pipeline sequence
    (prologue G2S + k-loop + epilogue) together, enabling removal of
    duplicate G2S stores.
    """

    def __init__(self):
        super().__init__()
        self._found = False

    def visit_seq_stmt_(self, op: SeqStmt) -> Stmt:
        if not self._found:
            stmts = list(op)
            # Check if this SeqStmt contains a pipeline loop
            has_loop = any(_is_pipeline_loop(s) for s in stmts)
            if has_loop:
                result = _transform_pipeline_seq(stmts)
                if result is not None:
                    self._found = True
                    return SeqStmt(result) if len(result) > 1 else result[0]

        # Default: recurse into children
        new_stmts = [self.visit_stmt(s) for s in op]
        if all(n.same_as(o) for n, o in zip(new_stmts, op)):
            return op
        return SeqStmt(new_stmts)

    def visit_for_(self, op: For) -> Stmt:
        # Still need to recurse into For nodes to find nested SeqStmts
        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op
        return For(
            op.loop_var, op.min, op.extent, op.kind, new_body,
            op.thread_binding, op.annotations,
        )


def InjectCdna4Pipeline():
    """TVM Pass: CDNA4 optimized pipeline with interleaved G2S and direct vmcnt."""

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = InjectCdna4PipelineMutator()
        new_body = mutator.visit_stmt(func.body)
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

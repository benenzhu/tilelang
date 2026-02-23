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
    """Wrap a G2S store in async_scope only (NO async_commit_queue_scope).

    This ensures InjectPTXAsyncCopy will convert it to ptx_cp_async
    but won't insert tvm_storage_sync between adjacent stores.
    """
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
    """Insert G2S loads into one MFMA stage, with optional vmcnt before MFMA.

    G2S stores are wrapped in async_scope only (not async_commit_queue_scope).
    """
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
    # S2R loads (sched_barrier + ds_reads)
    result.extend(stage_stmts[:insert_idx])
    # G2S loads (raw async_scope, no commit)
    for g in g2s_stores:
        result.append(_make_raw_g2s(g))
    # Second sched_barrier
    result.append(stage_stmts[insert_idx])
    # vmcnt before MFMA if specified
    if vmcnt_value is not None:
        result.append(_make_vmcnt(vmcnt_value))
    # s_barrier after vmcnt if requested (ensures all threads' G2S are visible)
    if barrier_after:
        result.append(_make_s_barrier())
    # MFMA block (setprio + MFMAs + setprio)
    result.extend(stage_stmts[insert_idx + 1:])

    return result


def _try_cdna4_pipeline(loop: For) -> Stmt | None:
    """Transform pipeline loop into CDNA4 interleaved schedule.

    1-step-ahead: G2S loads for k+1 are distributed across 4 MFMA stages
    of the current k iteration. vmcnt(0) + s_barrier at the last stage
    ensures all G2S complete before next iteration reads shared memory.
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

    # 2. Build stages — stage-reorder only (no G2S interleaving)
    # Keep original G2S + async_wait, just replace consumer with reordered stages
    # 2. Build interleaved stages with G2S distributed across them
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

    reordered_consumer = SeqStmt(phase_stmts)
    if alloc_wrappers:
        reordered_consumer = _rewrap(reordered_consumer, alloc_wrappers)

    # 3. Keep original G2S blocks and async_wait, replace only the consumer body.
    # This preserves the async_wait(inflight=1) → cp_async_wait<N> synchronization
    # that ensures the PREVIOUS k's G2S is complete before S2R reads shared memory.
    # The new G2S stores (interleaved into stages) will be for the NEXT k+1,
    # so they coexist with the original G2S which loads k+1 via the standard pipeline.
    new_body_stmts = []
    for stmt in body_stmts:
        if isinstance(stmt, AttrStmt) and stmt.attr_key in ("async_scope", "async_commit_queue_scope"):
            new_body_stmts.append(stmt)  # keep original G2S
        elif isinstance(stmt, AttrStmt) and stmt.attr_key == "async_wait_queue_scope":
            # Replace consumer inside async_wait with interleaved version
            wait_body = stmt.body
            if isinstance(wait_body, AttrStmt) and wait_body.attr_key == "async_wait_inflight_count":
                wrapped = AttrStmt(
                    wait_body.node, wait_body.attr_key, wait_body.value,
                    reordered_consumer,
                )
                new_body_stmts.append(AttrStmt(
                    stmt.node, stmt.attr_key, stmt.value, wrapped,
                ))
            else:
                new_body_stmts.append(stmt)

    new_body = SeqStmt(new_body_stmts) if len(new_body_stmts) > 1 else new_body_stmts[0]
    return For(
        k, loop.min, loop.extent,
        loop.kind, new_body, loop.thread_binding, loop.annotations,
    )


@tir.functor.mutator
class InjectCdna4PipelineMutator(tir.PyStmtExprMutator):
    """Find the pipeline k-loop and apply CDNA4 interleaved schedule."""

    def __init__(self):
        super().__init__()
        self._found = False

    def visit_for_(self, op: For) -> Stmt:
        if (
            not self._found
            and op.kind == ForKind.SERIAL
            and isinstance(op.extent, IntImm)
            and int(op.extent) > 1
        ):
            body_stmts = _flatten_seq(op.body)
            has_async = any(
                isinstance(s, AttrStmt)
                and s.attr_key in ("async_scope", "async_commit_queue_scope")
                for s in body_stmts
            )
            if has_async:
                result = _try_cdna4_pipeline(op)
                if result is not None:
                    self._found = True
                    return result

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

"""CDNA4 (gfx950) optimized pipeline: 2-step-ahead G2S with vmcnt(6).

Runs AFTER InjectSoftwarePipeline + LowerOpaqueBlock + Simplify.
Transforms the standard pipeline (all G2S → wait → compute) into
a HipKittens-style interleaved schedule:

  Phase 0: S2R + G2S(As[(k+1)&1][1], completing k+1) + MFMA
  Phase 1: S2R + G2S(As[k&1][0], starting k+2)       + MFMA
  Phase 2: S2R + G2S(Bs[k&1][0], k+2)                 + MFMA
  Phase 3: S2R + G2S(Bs[k&1][1], k+2) + vmcnt(6)      + MFMA

Uses direct `tl_waitcnt_vmcnt(N)` calls, bypassing InjectPTXAsyncCopy's
commit-group vmcnt mapping.
"""

from __future__ import annotations

from tvm import tir
from tvm.tir import (
    Allocate,
    AttrStmt,
    DeclBuffer,
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
    _is_setprio,
    _find_async_blocks,
    _find_consumer,
    _extract_consumer_stages,
    _make_async_g2s,
    _rewrap,
)


def _make_vmcnt(n: int) -> Stmt:
    """Emit asm volatile s_waitcnt vmcnt(N) directly."""
    return Evaluate(
        tir.call_extern("int32", "tl_waitcnt_vmcnt", IntImm("int32", n))
    )


def _interleave_one_stage(
    stage_stmts: list[Stmt],
    g2s_stores: list[Stmt],
    vmcnt_value: int | None = None,
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
    # S2R loads
    result.extend(stage_stmts[:insert_idx])
    # G2S loads
    for g in g2s_stores:
        result.append(_make_async_g2s(g))
    # Second sched_barrier
    result.append(stage_stmts[insert_idx])
    # vmcnt before MFMA if specified
    if vmcnt_value is not None:
        result.append(_make_vmcnt(vmcnt_value))
    # MFMA block (setprio + MFMAs + setprio)
    result.extend(stage_stmts[insert_idx + 1:])

    return result


def _try_cdna4_pipeline(loop: For) -> Stmt | None:
    """Transform pipeline loop into CDNA4 2-step-ahead schedule."""
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

    if len(mfma_stages) != 4 or len(g2s_a) != 4 or len(g2s_b) != 4:
        return None

    one = IntImm(k.dtype, 1)
    orig_n = int(loop.extent) if isinstance(loop.extent, IntImm) else None
    if orig_n is None or orig_n < 3:
        return None

    # 2. Build 2-step-ahead interleaved body
    # Phase 0: G2S A[2,3] for k+1 (completion), Phase 1-3: G2S for k+2
    phase_stmts = []
    phase_stmts.extend(_interleave_one_stage(
        mfma_stages[0], [g2s_a[2], g2s_a[3]],
    ))
    phase_stmts.extend(_interleave_one_stage(
        mfma_stages[1],
        [substitute(g2s_a[0], {k: k + one}), substitute(g2s_a[1], {k: k + one})],
    ))
    phase_stmts.extend(_interleave_one_stage(
        mfma_stages[2],
        [substitute(g2s_b[0], {k: k + one}), substitute(g2s_b[1], {k: k + one})],
    ))
    phase_stmts.extend(_interleave_one_stage(
        mfma_stages[3],
        [substitute(g2s_b[2], {k: k + one}), substitute(g2s_b[3], {k: k + one})],
        vmcnt_value=6,
    ))

    interleaved = SeqStmt(phase_stmts)
    if alloc_wrappers:
        interleaved = _rewrap(interleaved, alloc_wrappers)

    # 3. Main loop: k=0..N-3 (avoid k+2 OOB)
    main_loop = For(
        k, loop.min, IntImm(loop.extent.dtype, orig_n - 1),
        loop.kind, interleaved, loop.thread_binding, loop.annotations,
    )

    # 4. Last iter (k=N-2): standard 8 G2S for k+1, vmcnt(0)
    last_k = IntImm(k.dtype, orig_n - 2)
    all_g2s = g2s_a + g2s_b
    last_stmts = []
    for i, stage in enumerate(mfma_stages):
        g_start = i * 2
        g_end = g_start + 2
        last_stmts.extend(_interleave_one_stage(
            [substitute(s, {k: last_k}) for s in stage],
            [substitute(s, {k: last_k}) for s in all_g2s[g_start:g_end]],
            vmcnt_value=0 if i == 3 else None,
        ))
    last_body = SeqStmt(last_stmts)
    if alloc_wrappers:
        last_body = _rewrap(last_body, alloc_wrappers)

    # 5. Extra prologue: 6 G2S for k=1 (A[0,1] + B[0..3])
    k0 = IntImm(k.dtype, 0)
    extra = [substitute(s, {k: k0}) for s in [
        g2s_a[0], g2s_a[1], g2s_b[0], g2s_b[1], g2s_b[2], g2s_b[3],
    ]]
    extra_prologue = SeqStmt([_make_async_g2s(s) for s in extra])

    # 6. Prologue wait
    prologue_wait = _make_vmcnt(0)

    return SeqStmt([extra_prologue, prologue_wait, main_loop, last_body])


@tir.functor.mutator
class InjectCdna4PipelineMutator(tir.PyStmtExprMutator):
    """Find the pipeline k-loop and apply CDNA4 2-step-ahead schedule."""

    def __init__(self):
        super().__init__()
        self._found = False

    def visit_for_(self, op: For) -> Stmt:
        if (
            not self._found
            and op.kind == ForKind.SERIAL
            and isinstance(op.extent, IntImm)
            and int(op.extent) > 2
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
    """TVM Pass: CDNA4 optimized pipeline with 2-step-ahead G2S prefetch."""

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = InjectCdna4PipelineMutator()
        new_body = mutator.visit_stmt(func.body)
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

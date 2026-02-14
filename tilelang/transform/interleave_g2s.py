"""Interleave G2S (Global->Shared) loads with MFMA compute stages.

This pass restructures the pipeline loop body from a "load-all-then-compute-all"
pattern into a HipKittens-style interleaved pattern where G2S loads are issued
between MFMA phases, enabling overlap of memory and compute.

Before (current TileLang):
  G2S_A[0..3], G2S_B[0..3] -> commit -> wait(all) -> 4 MFMA stages

After (interleaved):
  stage_0(S2R + G2S[0,1] + commit + MFMA) ->
  stage_1(S2R + G2S[2,3] + commit + MFMA) ->
  stage_2(S2R + G2S[4,5] + commit + MFMA) ->
  stage_3(S2R + G2S[6,7] + commit + wait(0) + MFMA)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_seq(stmt: Stmt) -> list[Stmt]:
    """Flatten nested SeqStmt into a flat list of statements."""
    if isinstance(stmt, SeqStmt):
        result = []
        for s in stmt:
            result.extend(_flatten_seq(s))
        return result
    return [stmt]


def _is_call_extern(stmt: Stmt, name: str) -> bool:
    """Check if stmt is Evaluate(call_extern(..., name, ...))."""
    if not isinstance(stmt, Evaluate):
        return False
    call = stmt.value
    if not isinstance(call, tir.Call):
        return False
    if call.op.name != "tir.call_extern":
        return False
    if len(call.args) < 1:
        return False
    arg0 = call.args[0]
    if isinstance(arg0, tir.StringImm) and arg0.value == name:
        return True
    return False


def _is_sched_barrier(stmt: Stmt) -> bool:
    return _is_call_extern(stmt, "__builtin_amdgcn_sched_barrier")


def _is_setprio(stmt: Stmt, val: int | None = None) -> bool:
    if not _is_call_extern(stmt, "__builtin_amdgcn_s_setprio"):
        return False
    if val is not None:
        call = stmt.value
        arg1 = call.args[1]
        if isinstance(arg1, IntImm) and arg1.value == val:
            return True
        return False
    return True


def _make_async_g2s(inner_store_loop: Stmt) -> Stmt:
    """Wrap a single G2S vectorized store loop in async_commit(async_scope(...))."""
    scope_attr = AttrStmt(
        IntImm("int32", 0), "async_scope", IntImm("int32", 1),
        inner_store_loop,
    )
    return AttrStmt(
        IntImm("int32", 0), "async_commit_queue_scope", IntImm("int32", 0),
        scope_attr,
    )


def _make_async_wait_wrapping(inflight: int, body: Stmt) -> Stmt:
    """Wrap body in async_wait with the given inflight count."""
    inner = AttrStmt(
        IntImm("int32", 0), "async_wait_inflight_count", IntImm("int32", inflight),
        body,
    )
    return AttrStmt(
        IntImm("int32", 0), "async_wait_queue_scope", IntImm("int32", 0),
        inner,
    )


# ---------------------------------------------------------------------------
# G2S extraction
# ---------------------------------------------------------------------------


def _extract_g2s_from_async_block(attr_stmt: AttrStmt) -> list[Stmt]:
    """Extract individual G2S stores from an async block.

    Splits the unrolled loop into individual vectorized-loop stores.
    """
    body = attr_stmt.body
    while isinstance(body, AttrStmt):
        body = body.body

    if not isinstance(body, For):
        return [body]

    unroll_loop = body
    if unroll_loop.kind != ForKind.UNROLLED:
        return [unroll_loop]

    loop_var = unroll_loop.loop_var
    extent = int(unroll_loop.extent)
    inner_body = unroll_loop.body

    while isinstance(inner_body, AttrStmt) and inner_body.attr_key == "pragma_unroll_explicit":
        inner_body = inner_body.body

    individual_stores = []
    for i in range(extent):
        substituted = substitute(inner_body, {loop_var: IntImm(loop_var.dtype, i)})
        individual_stores.append(substituted)

    return individual_stores


def _find_async_blocks(stmts: list[Stmt]):
    """Find G2S async blocks in a list of statements.

    Returns (g2s_a_stores, g2s_b_stores, remaining_stmts).
    """
    g2s_blocks = []
    remaining = []

    for stmt in stmts:
        if isinstance(stmt, AttrStmt) and stmt.attr_key in ("async_scope", "async_commit_queue_scope"):
            g2s_blocks.append(stmt)
        else:
            remaining.append(stmt)

    if len(g2s_blocks) < 2:
        return None, None, stmts

    g2s_a = _extract_g2s_from_async_block(g2s_blocks[0])
    g2s_b = _extract_g2s_from_async_block(g2s_blocks[1])

    return g2s_a, g2s_b, remaining


# ---------------------------------------------------------------------------
# MFMA stage extraction
# ---------------------------------------------------------------------------


def _unwrap_alloc_decl(body: Stmt):
    """Unwrap Allocate/DeclBuffer/AttrStmt wrappers to get inner body."""
    wrappers = []
    while True:
        if isinstance(body, DeclBuffer):
            wrappers.append(("DeclBuffer", body.buffer))
            body = body.body
        elif isinstance(body, Allocate):
            wrappers.append(("Allocate", (body.buffer_var, body.dtype, body.extents,
                                          body.condition, body.annotations)))
            body = body.body
        elif isinstance(body, AttrStmt) and body.attr_key == "pragma_unroll_explicit":
            wrappers.append(("AttrStmt", (body.node, body.attr_key, body.value)))
            body = body.body
        else:
            break
    return body, wrappers


def _rewrap(body: Stmt, wrappers: list) -> Stmt:
    """Re-apply wrappers around a body in reverse order."""
    for kind, args in reversed(wrappers):
        if kind == "DeclBuffer":
            body = DeclBuffer(args, body)
        elif kind == "Allocate":
            buf_var, dtype, extents, cond, annot = args
            body = Allocate(buf_var, dtype, extents, cond, body, annot)
        elif kind == "AttrStmt":
            node, key, val = args
            body = AttrStmt(node, key, val, body)
    return body


def _split_into_stages(stmts: list[Stmt]) -> list[list[Stmt]] | None:
    """Split a flat list of statements into MFMA stages.

    Each stage: sched_barrier -> S2R loads -> sched_barrier -> setprio(1) -> MFMAs -> setprio(0)
    """
    stages = []
    current_stage = []
    in_stage = False

    for stmt in stmts:
        if _is_sched_barrier(stmt) and not in_stage:
            in_stage = True
            current_stage = [stmt]
        elif in_stage:
            current_stage.append(stmt)
            if _is_setprio(stmt, 0):
                stages.append(current_stage)
                current_stage = []
                in_stage = False
        else:
            if stages:
                stages[-1].append(stmt)
            else:
                current_stage.append(stmt)

    if not stages and current_stage:
        return None
    return stages


def _extract_consumer_stages(consumer_body: Stmt):
    """Extract MFMA stages from the consumer block.

    Handles both:
    - ki-loop consumer (k_pack=1): For(ki, ...) containing stages
    - Direct consumer (k_pack=2): stages directly in the body

    Returns (stages_list, alloc_wrappers) or None on failure.
    """
    inner_body, wrappers = _unwrap_alloc_decl(consumer_body)

    # Check if inner_body is a For loop (ki-loop case)
    if isinstance(inner_body, For) and inner_body.kind == ForKind.SERIAL:
        ki_loop = inner_body
        loop_var = ki_loop.loop_var
        extent = int(ki_loop.extent)
        ki_inner, ki_wrappers = _unwrap_alloc_decl(ki_loop.body)

        all_stages = []
        for ki_val in range(extent):
            unrolled = substitute(ki_inner, {loop_var: IntImm(loop_var.dtype, ki_val)})
            stmts = _flatten_seq(unrolled)
            stages = _split_into_stages(stmts)
            if stages is None:
                return None
            all_stages.extend(stages)

        # Merge both wrapper levels
        all_wrappers = wrappers + ki_wrappers
        return all_stages, all_wrappers

    # Direct case (no ki loop) — stages directly in inner_body
    stmts = _flatten_seq(inner_body)
    stages = _split_into_stages(stmts)
    if stages is None:
        return None
    return stages, wrappers


# ---------------------------------------------------------------------------
# Find consumer
# ---------------------------------------------------------------------------


def _find_consumer(stmts: list[Stmt]):
    """Find the consumer block wrapped in async_wait.

    Returns (consumer_body, found) where consumer_body is the body
    inside the async_wait wrapper.
    """
    for stmt in stmts:
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "async_wait_queue_scope":
            body = stmt.body
            if isinstance(body, AttrStmt) and body.attr_key == "async_wait_inflight_count":
                return body.body, True
    # Fallback: look for any serial For
    for stmt in stmts:
        if isinstance(stmt, For) and stmt.kind == ForKind.SERIAL:
            return stmt, False
    return None, False


# ---------------------------------------------------------------------------
# Build interleaved body
# ---------------------------------------------------------------------------


def _build_interleaved_body(
    g2s_stores: list[Stmt],
    mfma_stages: list[list[Stmt]],
) -> Stmt:
    """Build the interleaved loop body.

    Distributes G2S loads evenly across MFMA stages.
    Inserts vmcnt(0) + s_barrier at the last stage to ensure all
    G2S loads complete before the next iteration reads shared memory.
    """
    num_stages = len(mfma_stages)
    num_g2s = len(g2s_stores)
    # How many G2S per stage (may be > 1 if 8 G2S / 4 stages)
    g2s_per_stage = (num_g2s + num_stages - 1) // num_stages

    all_stmts = []

    for i, stage_stmts in enumerate(mfma_stages):
        # Find insertion point: before the second sched_barrier
        sched_barrier_count = 0
        insert_idx = None

        for j, s in enumerate(stage_stmts):
            if _is_sched_barrier(s):
                sched_barrier_count += 1
                if sched_barrier_count == 2:
                    insert_idx = j
                    break

        if insert_idx is None:
            all_stmts.extend(stage_stmts)
            continue

        new_stage = []

        # Part 1: S2R loads (up to second sched_barrier)
        new_stage.extend(stage_stmts[:insert_idx])

        # Part 2: G2S loads for this stage
        g2s_start = i * g2s_per_stage
        g2s_end = min(g2s_start + g2s_per_stage, num_g2s)
        for gi in range(g2s_start, g2s_end):
            new_stage.append(_make_async_g2s(g2s_stores[gi]))

        # Part 3: Second sched_barrier
        new_stage.append(stage_stmts[insert_idx])

        # Part 4: MFMA block, last stage gets vmcnt(0) wait
        remaining_stmts = stage_stmts[insert_idx + 1:]
        if i == num_stages - 1:
            # Last stage: wrap MFMA in async_wait(0) to drain all G2S
            remaining_body = SeqStmt(remaining_stmts) if len(remaining_stmts) > 1 else remaining_stmts[0]
            wait_wrapped = _make_async_wait_wrapping(0, remaining_body)
            new_stage.append(wait_wrapped)
        else:
            new_stage.extend(remaining_stmts)

        all_stmts.extend(new_stage)

    return SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]


# ---------------------------------------------------------------------------
# Main pass logic
# ---------------------------------------------------------------------------


def _try_interleave_loop_body(loop: For) -> Stmt | None:
    """Try to apply G2S/MFMA interleaving to a pipeline k-loop.

    Returns a SeqStmt containing:
      1. A prologue async_wait(0) to drain the initial G2S loads
      2. The transformed loop with interleaved G2S/MFMA
    """
    body_stmts = _flatten_seq(loop.body)

    # 1. Find G2S async blocks
    g2s_a, g2s_b, remaining = _find_async_blocks(body_stmts)
    if g2s_a is None or g2s_b is None:
        return None

    # 2. Find consumer block
    consumer_body, _found = _find_consumer(remaining)
    if consumer_body is None:
        return None

    # 3. Extract MFMA stages
    result = _extract_consumer_stages(consumer_body)
    if result is None:
        return None
    mfma_stages, alloc_wrappers = result
    if len(mfma_stages) == 0:
        return None

    # 4. Combine all G2S stores
    all_g2s = g2s_a + g2s_b
    if len(all_g2s) == 0 or len(mfma_stages) == 0:
        return None

    # 5. Build interleaved body (no per-iteration prologue wait)
    interleaved = _build_interleaved_body(all_g2s, mfma_stages)

    # 6. Wrap with allocations
    if alloc_wrappers:
        interleaved = _rewrap(interleaved, alloc_wrappers)

    # 7. Build the new loop
    new_loop = For(
        loop.loop_var, loop.min, loop.extent, loop.kind,
        interleaved, loop.thread_binding, loop.annotations,
    )

    # 8. Prologue wait: drain the initial G2S loads ONCE before the loop.
    #    Wrap an s_barrier so InjectPTXAsyncCopy doesn't drop it.
    barrier_call = Evaluate(
        tir.call_extern("int32", "__builtin_amdgcn_s_barrier"),
    )
    prologue_wait = _make_async_wait_wrapping(0, barrier_call)

    return SeqStmt([prologue_wait, new_loop])


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------


@tir.functor.mutator
class InterleaveG2SMutator(tir.PyStmtExprMutator):
    """Mutator that finds the pipeline k-loop and interleaves G2S with MFMA."""

    def __init__(self):
        super().__init__()
        self._found_pipeline_loop = False

    def visit_for_(self, op: For) -> Stmt:
        # Only transform the first serial loop with extent > 1 (the pipeline k-loop)
        if (
            not self._found_pipeline_loop
            and op.kind == ForKind.SERIAL
            and isinstance(op.extent, IntImm)
            and int(op.extent) > 1
        ):
            # Check if this looks like a pipeline loop (has async_scope children)
            body_stmts = _flatten_seq(op.body)
            has_async = any(
                isinstance(s, AttrStmt) and s.attr_key in ("async_scope", "async_commit_queue_scope")
                for s in body_stmts
            )
            if has_async:
                result = _try_interleave_loop_body(op)
                if result is not None:
                    self._found_pipeline_loop = True
                    return result

        # Default: recurse into children
        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op
        return For(
            op.loop_var, op.min, op.extent, op.kind, new_body,
            op.thread_binding, op.annotations,
        )


def InterleaveG2SWithCompute():
    """TVM Pass: Interleave G2S loads with MFMA compute stages.

    Restructures the pipeline loop body so G2S loads for the next
    k-iteration are distributed across MFMA compute phases rather
    than batched before compute. Uses vmcnt(0) at the last stage
    to ensure all G2S loads complete before the next iteration.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = InterleaveG2SMutator()
        new_body = mutator.visit_stmt(func.body)
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

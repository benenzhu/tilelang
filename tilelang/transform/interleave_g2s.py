"""Interleave G2S copies with 4-cluster MFMA compute.

After the pingpong 4-cluster compute schedule and software pipelining,
the main k-loop has G2S copies at the top and compute at the bottom.
This pass distributes the G2S copies across the 4 compute clusters:

Before:
    sync; G2S_A(4); sync; G2S_B(4); commit; wait<8>; sync; compute(4 clusters)

After:
    wait<8>; sync; compute_with_interleaved_G2S; commit

Where compute_with_interleaved_G2S (per ki iteration):
    Cluster 0: ldA_sub0, ldB_sub0, [ki==0: G2S_A[0:2]], mfma(0,0)
    Cluster 1: ldB_sub1,           [ki==0: G2S_A[2:4]], mfma(0,1)
    Cluster 2: ldA_sub1,           [ki==0: G2S_B[0:2]], mfma(1,0)
    Cluster 3:                     [ki==0: G2S_B[2:4]], mfma(1,1)
"""

from tvm import tir
from tvm.tir import (
    AttrStmt,
    Evaluate,
    For,
    IfThenElse,
    PrimFunc,
    SeqStmt,
    stmt_functor,
)
from tvm.tir.transform import prim_func_pass

from tilelang.utils.target import target_is_gfx950

_op_ptx_cp_async_lds_rsrc = tir.op.Op.get("tl.ptx_cp_async_lds_rsrc")


def _is_async_copy_call(stmt):
    """Check if stmt is an Evaluate(Call(ptx_cp_async_lds_rsrc(...)))."""
    if isinstance(stmt, Evaluate) and isinstance(stmt.value, tir.Call):
        return stmt.value.op == _op_ptx_cp_async_lds_rsrc
    return False


def _is_storage_sync(stmt, scope=None):
    """Check if stmt is tvm_storage_sync(scope)."""
    if isinstance(stmt, Evaluate) and isinstance(stmt.value, tir.Call):
        if stmt.value.op == tir.op.Op.get("tir.tvm_storage_sync"):
            if scope is None:
                return True
            if len(stmt.value.args) > 0 and isinstance(stmt.value.args[0], tir.StringImm):
                return stmt.value.args[0].value == scope
    return False


def _is_for_with_async_copy(stmt):
    """Check if stmt is a For loop containing async copy calls."""
    if not isinstance(stmt, For):
        return False
    found = [False]

    def _v(s):
        if _is_async_copy_call(s):
            found[0] = True

    stmt_functor.post_order_visit(stmt.body, _v)
    return found[0]


def _split_for_half(for_node):
    """Split a For(var, 0, N, ...) into For(var, 0, N//2, ...) and For(var, N//2, N//2, ...)."""
    assert isinstance(for_node.extent, tir.IntImm)
    n = for_node.extent.value
    half = n // 2
    first = For(for_node.loop_var, tir.IntImm("int32", 0),
                tir.IntImm("int32", half), for_node.kind,
                for_node.body, for_node.thread_binding, for_node.annotations)
    second = For(for_node.loop_var, tir.IntImm("int32", half),
                 tir.IntImm("int32", half), for_node.kind,
                 for_node.body, for_node.thread_binding, for_node.annotations)
    return first, second


def _parse_k_loop_body(seq):
    """Parse the k-loop body SeqStmt to extract G2S and compute sections.

    Expected pattern (SeqStmt elements):
      [0] sync("shared.dyn")
      [1] For(unroll): G2S_A copies
      [2] AttrStmt(async_commit_queue_scope):
            sync("shared.dyn")
            For(unroll): G2S_B copies
      [3] AttrStmt(async_wait_queue_scope):
            AttrStmt(async_wait_inflight_count):
              sync("shared")
              For(ki, serial): compute

    Returns (g2s_a_loop, g2s_b_loop, wait_count, sync_before_compute, compute_loop) or None.
    """
    if not isinstance(seq, SeqStmt) or len(seq.seq) < 4:
        return None

    stmts = list(seq.seq)

    # Find G2S_A: first For loop with async copies
    g2s_a = None
    for s in stmts:
        if _is_for_with_async_copy(s):
            g2s_a = s
            break

    # Find commit scope containing G2S_B
    g2s_b = None
    commit_scope = None
    for s in stmts:
        if isinstance(s, AttrStmt) and s.attr_key == "async_commit_queue_scope":
            commit_scope = s
            # Inside commit scope, find the For loop with async copies
            body = s.body
            if isinstance(body, SeqStmt):
                for inner in body.seq:
                    if _is_for_with_async_copy(inner):
                        g2s_b = inner
                        break
            elif _is_for_with_async_copy(body):
                g2s_b = body
            break

    # Find wait scope containing compute
    wait_count = 0
    sync_before_compute = None
    compute_loop = None
    for s in stmts:
        if isinstance(s, AttrStmt) and s.attr_key == "async_wait_queue_scope":
            inner = s.body
            if isinstance(inner, AttrStmt) and inner.attr_key == "async_wait_inflight_count":
                if isinstance(inner.value, tir.IntImm):
                    wait_count = inner.value.value
                # Parse inner body for sync + compute
                wait_body = inner.body
                if isinstance(wait_body, SeqStmt):
                    for ws in wait_body.seq:
                        if _is_storage_sync(ws, "shared"):
                            sync_before_compute = ws
                        elif isinstance(ws, For):
                            compute_loop = ws
                elif isinstance(wait_body, For):
                    compute_loop = wait_body
            break

    if g2s_a is None or g2s_b is None or compute_loop is None:
        return None

    return g2s_a, g2s_b, wait_count, sync_before_compute, compute_loop


def _interleave_g2s_into_compute(compute_loop, g2s_groups):
    """Insert G2S sub-loops between compute clusters, guarded by if ki==0.

    compute_loop body per ki iteration has 8 statements (4 clusters × 2):
      [0] ldA_sub0      [1] ldB_sub0      [2] mfma(0,0)
      [3] ldB_sub1      [4] mfma(0,1)
      [5] ldA_sub1      [6] mfma(1,0)
      [7] mfma(1,1)

    Insert G2S before each cluster's mfma (indices 2, 4, 6, 7):
      [0] ldA_sub0  [1] ldB_sub0  [NEW] if(ki==0): G2S_A[0:2]  [2] mfma(0,0)
      [3] ldB_sub1  [NEW] if(ki==0): G2S_A[2:4]  [4] mfma(0,1)
      [5] ldA_sub1  [NEW] if(ki==0): G2S_B[0:2]  [6] mfma(1,0)
      [NEW] if(ki==0): G2S_B[2:4]  [7] mfma(1,1)
    """
    ki_var = compute_loop.loop_var
    body = compute_loop.body

    if not isinstance(body, SeqStmt) or len(body.seq) != 8:
        return None

    stmts = list(body.seq)
    # g2s_groups = [g2s_a_0, g2s_a_1, g2s_b_0, g2s_b_1]
    # Insert positions: before stmts[2], before stmts[4], before stmts[6], before stmts[7]
    guard = lambda g2s: IfThenElse(
        tir.EQ(ki_var, tir.IntImm("int32", 0)),
        g2s,
        None,
    )

    new_stmts = [
        stmts[0],              # ldA_sub0
        stmts[1],              # ldB_sub0
        guard(g2s_groups[0]),   # G2S_A[0:2]
        stmts[2],              # mfma(0,0)
        stmts[3],              # ldB_sub1
        guard(g2s_groups[1]),   # G2S_A[2:4]
        stmts[4],              # mfma(0,1)
        stmts[5],              # ldA_sub1
        guard(g2s_groups[2]),   # G2S_B[0:2]
        stmts[6],              # mfma(1,0)
        guard(g2s_groups[3]),   # G2S_B[2:4]
        stmts[7],              # mfma(1,1)
    ]

    new_body = SeqStmt(new_stmts)
    return For(compute_loop.loop_var, compute_loop.min, compute_loop.extent,
               compute_loop.kind, new_body,
               compute_loop.thread_binding, compute_loop.annotations)


def _rebuild_k_loop(k_loop, wait_count, sync_before_compute,
                    new_compute_loop, commit_stmt):
    """Build new k-loop body: wait → sync → compute(with G2S) → commit."""
    parts = []

    # wait
    wait_inner = SeqStmt([sync_before_compute, new_compute_loop]) if sync_before_compute else new_compute_loop
    wait_attr = AttrStmt(
        tir.IntImm("int32", 0), "async_wait_inflight_count",
        tir.IntImm("int32", wait_count), wait_inner)
    wait_scope = AttrStmt(
        tir.IntImm("int32", 0), "async_wait_queue_scope",
        tir.IntImm("int32", 0), wait_attr)
    parts.append(wait_scope)

    # commit
    parts.append(commit_stmt)

    new_body = SeqStmt(parts)
    return For(k_loop.loop_var, k_loop.min, k_loop.extent,
               k_loop.kind, new_body,
               k_loop.thread_binding, k_loop.annotations)


class _ForReplacer:
    """Replace a For loop identified by its loop_var in the IR tree.

    Uses a stateful class to track whether replacement happened,
    avoiding unnecessary reconstruction of untouched subtrees.
    """

    def __init__(self, target_var, replacement):
        self.target_var = target_var
        self.replacement = replacement
        self.done = False

    def visit(self, stmt):
        if self.done:
            return stmt
        if isinstance(stmt, For) and stmt.loop_var.same_as(self.target_var):
            self.done = True
            return self.replacement
        if isinstance(stmt, SeqStmt):
            new_seq = []
            found = False
            for s in stmt.seq:
                if found:
                    new_seq.append(s)  # don't visit after replacement
                else:
                    ns = self.visit(s)
                    new_seq.append(ns)
                    if self.done:
                        found = True
            return SeqStmt(new_seq) if self.done else stmt
        if isinstance(stmt, AttrStmt):
            new_body = self.visit(stmt.body)
            return AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body) if self.done else stmt
        if isinstance(stmt, tir.LetStmt):
            new_body = self.visit(stmt.body)
            return tir.LetStmt(stmt.var, stmt.value, new_body) if self.done else stmt
        if isinstance(stmt, For):
            new_body = self.visit(stmt.body)
            if self.done:
                return For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                           new_body, stmt.thread_binding, stmt.annotations)
            return stmt
        if isinstance(stmt, tir.DeclBuffer):
            new_body = self.visit(stmt.body)
            return tir.DeclBuffer(stmt.buffer, new_body) if self.done else stmt
        if isinstance(stmt, tir.Allocate):
            new_body = self.visit(stmt.body)
            if self.done:
                return tir.Allocate(stmt.buffer_var, stmt.dtype, stmt.extents,
                                    stmt.condition, new_body, stmt.annotations)
            return stmt
        return stmt


def _replace_for_in_tree(stmt, target_var, replacement):
    return _ForReplacer(target_var, replacement).visit(stmt)


def InterleaveG2SMfma():
    """Interleave G2S copies with 4-cluster MFMA compute for gfx950."""

    def pass_fn(func: PrimFunc, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func

        # Find the main k-loop
        from tilelang.transform.hoist_buffer_resource import _find_for_with_commit
        k_loop = _find_for_with_commit(func.body)
        if k_loop is None:
            return func

        # Parse k-loop body
        parsed = _parse_k_loop_body(k_loop.body)
        if parsed is None:
            return func

        g2s_a, g2s_b, wait_count, sync_before_compute, compute_loop = parsed

        # Verify compute loop has 8 statements per iteration (4-cluster pingpong)
        if not isinstance(compute_loop.body, SeqStmt) or len(compute_loop.body.seq) != 8:
            return func

        # Verify G2S loops have even extents (so we can split in half)
        if not isinstance(g2s_a.extent, tir.IntImm) or g2s_a.extent.value % 2 != 0:
            return func
        if not isinstance(g2s_b.extent, tir.IntImm) or g2s_b.extent.value % 2 != 0:
            return func

        # Split G2S loops into halves
        g2s_a_0, g2s_a_1 = _split_for_half(g2s_a)
        g2s_b_0, g2s_b_1 = _split_for_half(g2s_b)

        # Build interleaved compute loop
        new_compute = _interleave_g2s_into_compute(
            compute_loop, [g2s_a_0, g2s_a_1, g2s_b_0, g2s_b_1])
        if new_compute is None:
            return func

        # Build commit statement (async_commit_queue_scope wrapping empty body)
        commit_stmt = AttrStmt(
            tir.IntImm("int32", 0), "async_commit_queue_scope",
            tir.IntImm("int32", 0), Evaluate(tir.IntImm("int32", 0)))

        # Build new k-loop
        new_k_loop = _rebuild_k_loop(
            k_loop, wait_count, sync_before_compute,
            new_compute, commit_stmt)

        # Replace old k-loop with new one.
        # Use tir.stmt_functor.substitute which replaces Var references.
        # We substitute the k-loop's body by replacing it via ir_transform
        # on only the k-loop itself (small tree, no stack overflow).
        # Then find-and-replace the k-loop in the outer tree manually.
        k_loop_var = k_loop.loop_var
        new_body = _replace_for_in_tree(func.body, k_loop_var, new_k_loop)
        return func.with_body(new_body)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)

from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.mfma_macro_generator import (
    MatrixCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


def _is_gfx950() -> bool:
    """Detect whether the current device is GFX950 via torch."""
    try:
        import torch
        if torch.version.hip is None or not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return gcn_arch.startswith("gfx950")
    except Exception:
        return False


def _get_mfma_k_dim(target: Target, in_dtype: str) -> int | None:
    """Return the MFMA k-dimension override for GFX950, or None for default."""
    if target.kind.name != "hip":
        return None
    if not _is_gfx950():
        return None
    # GFX950 supports 16x16x32 MFMA for bf16 and f16
    if in_dtype in ("bfloat16", "float16"):
        return 32
    return None


class GemmMFMA(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MFMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mfma_emitter = MatrixCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            k_pack=self.k_pack,
            mfma_k_dim=_get_mfma_k_dim(target, self.in_dtype),
        )

        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mfma_emitter.make_mfma_load_layout(self.B, matrix="B"),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mfma_emitter.make_mfma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mfma_emitter.make_mfma_load_layout(self.A, matrix="A"),
                self.B: mfma_emitter.make_mfma_load_layout(self.B, matrix="B"),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MFMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mfma_emitter = MatrixCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            k_pack=self.k_pack,
            mfma_k_dim=_get_mfma_k_dim(target, self.in_dtype),
        )

        in_dtype = self.in_dtype
        warp_rows = mfma_emitter.warp_rows
        warp_cols = mfma_emitter.warp_cols
        local_size_a = mfma_emitter.local_size_a
        local_size_b = mfma_emitter.local_size_b
        block_K = mfma_emitter.chunk
        micro_size_k = mfma_emitter.micro_size_k
        # Use region for shared-memory operands if available
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():
            # Pre-compute C_local float32x4 offsets for setprio register deps
            # offset = row_idx * warp_cols (in float32x4 units)
            _c_off_row0 = 0 * warp_cols
            _c_off_row1 = 1 * warp_cols
            _c_off_row2 = 2 * warp_cols
            _c_off_row3 = 3 * warp_cols

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                Fully-unrolled 8-round S2R-interleaved GEMM (gemm_ss)
                with fine-grained scheduling barriers and priority hints.

                Schedule (per ki iteration):
                  Preamble  – 6 ds_read before first MFMA:
                    load A rows 0-1 (4), load B col 0 (2)
                  Round 0a  – MFMA rows 0-1 × col 0  (4 mfma)
                    load A rows 2-3 (4, hidden behind mfma)
                  Round 0b  – MFMA rows 2-3 × col 0  (4 mfma)
                  Rounds 1-7 – per col: load B col (2) + MFMA all rows (8)

                Scheduling annotations:
                  sched_barrier(0)  between every load/MFMA region boundary
                  asm volatile s_setprio with VGPR deps for rounds 0a/0b
                  __builtin_amdgcn_s_setprio for round 1
                  s_barrier() between round 0a→0b, 0b→1, 1→2
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)
                B_local = T.alloc_local((local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # ── sched_barrier: start of iteration ──
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── Preamble: 6 ds_read before first MFMA ──
                    mfma_emitter.ldmatrix_a_single(A_local, A_region, ki, T.int32(0))
                    mfma_emitter.ldmatrix_a_single(A_local, A_region, ki, T.int32(1))
                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(0))

                    # ── sched_barrier: end of preamble loads ──
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── s_setprio 1 with VGPR deps on C rows 0,1 ──
                    tir.call_extern("int32", "tl_setprio_hi", C_buf.data,
                                    tir.IntImm("int32", _c_off_row0),
                                    tir.IntImm("int32", _c_off_row1))

                    # ── Round 0a: MFMA rows 0-1 × col 0 (4 mfma) ──
                    mfma_emitter.mfma_cell(A_local, B_local, C_buf, ki, T.int32(0), T.int32(0))
                    mfma_emitter.mfma_cell(A_local, B_local, C_buf, ki, T.int32(1), T.int32(0))

                    # ── s_setprio 0 with VGPR deps on C rows 0,1 ──
                    tir.call_extern("int32", "tl_setprio_lo", C_buf.data,
                                    tir.IntImm("int32", _c_off_row0),
                                    tir.IntImm("int32", _c_off_row1))

                    # ── s_barrier + sched_barrier: transition to A rows 2-3 loads ──
                    tir.call_extern("int32", "__builtin_amdgcn_s_barrier")
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── Load A rows 2-3 with sched_barrier between them ──
                    mfma_emitter.ldmatrix_a_single(A_local, A_region, ki, T.int32(2))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.ldmatrix_a_single(A_local, A_region, ki, T.int32(3))

                    # ── sched_barrier: end of A rows 2-3 loads ──
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── s_setprio 1 with VGPR deps on C rows 2,3 ──
                    tir.call_extern("int32", "tl_setprio_hi", C_buf.data,
                                    tir.IntImm("int32", _c_off_row2),
                                    tir.IntImm("int32", _c_off_row3))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── Round 0b: MFMA rows 2-3 × col 0 (4 mfma) ──
                    mfma_emitter.mfma_cell(A_local, B_local, C_buf, ki, T.int32(2), T.int32(0))
                    mfma_emitter.mfma_cell(A_local, B_local, C_buf, ki, T.int32(3), T.int32(0))

                    # ── sched_barrier + s_setprio 0 ──
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    tir.call_extern("int32", "tl_setprio_lo", C_buf.data,
                                    tir.IntImm("int32", _c_off_row2),
                                    tir.IntImm("int32", _c_off_row3))

                    # ── s_barrier + sched_barrier: transition to round 1 ──
                    tir.call_extern("int32", "__builtin_amdgcn_s_barrier")
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── Round 1: B col 1 load + MFMA ──
                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(1))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    tir.call_extern("int32", "__builtin_amdgcn_s_setprio", tir.IntImm("int32", 1))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(1))
                    tir.call_extern("int32", "__builtin_amdgcn_s_setprio", tir.IntImm("int32", 0))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── s_barrier + sched_barrier: transition to rounds 2-7 ──
                    tir.call_extern("int32", "__builtin_amdgcn_s_barrier")
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    # ── Rounds 2-7: B col load + sched_barrier + MFMA + sched_barrier ──
                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(2))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(2))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(3))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(3))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(4))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(4))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(5))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(5))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(6))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(6))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))

                    mfma_emitter.ldmatrix_b_single(B_local, B_region, ki, T.int32(7))
                    tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", tir.IntImm("int32", 0))
                    mfma_emitter.mfma_col_slice(A_local, B_local, C_buf, ki, T.int32(7))

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffer A_shared
                into local fragments, then issues Matrix Core mfma ops with
                B already in registers, accumulating into C_local.

                S2R loads for A are interleaved with MFMA compute on a
                per-warp-row basis.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)

                if clear_accum:
                    T.clear(C_buf)

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load first row of A (ensures A_local declared
                    # before B in codegen -- workaround for HIP
                    # compiler VGPR allocation issue)
                    mfma_emitter.ldmatrix_a_single(
                        A_local,
                        A_region,
                        ki,
                        T.int32(0),
                    )
                    # MFMA for row 0
                    mfma_emitter.mfma_slice(A_local, B_buf, C_buf, ki, T.int32(0))
                    # Interleave: remaining rows of A + MFMA
                    for ri in T.serial(1, warp_rows):
                        mfma_emitter.ldmatrix_a_single(
                            A_local,
                            A_region,
                            ki,
                            ri,
                        )
                        mfma_emitter.mfma_slice(A_local, B_buf, C_buf, ki, ri)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffer B_shared
                into local fragments, with A already in registers, then issues
                Matrix Core mfma ops, accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mfma ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)

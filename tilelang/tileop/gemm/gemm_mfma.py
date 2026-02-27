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
import tilelang
from tilelang.transform import PassConfigKey


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
        print(f"m_warp: {m_warp}, n_warp: {n_warp}")
        warp_row_tiles = int(self.M // m_warp) # m_warp: 4, n_warp: 2 # 64
        warp_col_tiles = int(self.N // n_warp) # 128
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

        k_pack = mfma_emitter.k_pack
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

        assert block_K >= micro_size_k * k_pack, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k}) * k_pack ({k_pack})"
        assert block_K % (micro_size_k * k_pack) == 0, (
            f"block_K ({block_K}) must be divisible by micro_size_k ({micro_size_k}) * k_pack ({k_pack})"
        )

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():
            # Check if scattered warp layout is enabled via pass config
            pass_ctx = tilelang.transform.get_pass_context()
            use_scattered = bool(
                pass_ctx and pass_ctx.config.get(
                    PassConfigKey.TL_SCATTERED_WARP_LAYOUT, False))

            if use_scattered:
                # HipKittens-style scattered warp layout:
                # Split M and N into 2 halves each, iterate over 4 quadrants.
                # Each quadrant uses half the warp_rows and warp_cols.
                m_subtiles = 2
                n_subtiles = 2
                sub_warp_rows = warp_rows // m_subtiles
                sub_warp_cols = warp_cols // n_subtiles

                @T.prim_func
                def _gemm_ssr_scattered() -> None:
                    # Half-sized local buffers (only need to hold one quadrant's data)
                    A_local = T.alloc_local((sub_warp_rows * local_size_a * self.k_pack), in_dtype)
                    B_local = T.alloc_local((sub_warp_cols * local_size_b * self.k_pack), in_dtype)
                    if clear_accum:
                        T.clear(C_buf)
                    for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                        for sub_m, sub_n in T.grid(m_subtiles, n_subtiles):
                            # S2R A: load sub_warp_rows from A_shared[sub_m * half_M :]
                            mfma_emitter.ldmatrix_a(
                                A_local, A_region, ki,
                                num_rows=sub_warp_rows,
                                row_offset=sub_m * sub_warp_rows,
                            )
                            # S2R B: load sub_warp_cols from B_shared[sub_n * half_N :]
                            mfma_emitter.ldmatrix_b(
                                B_local, B_region, ki,
                                num_cols=sub_warp_cols,
                                col_offset=sub_n * sub_warp_cols,
                            )
                            # MFMA: compute sub-block, write to correct C_local position
                            mfma_emitter.mfma(
                                A_local, B_local, C_buf, ki,
                                num_rows=sub_warp_rows,
                                num_cols=sub_warp_cols,
                                c_row_offset=sub_m * sub_warp_rows,
                                c_col_offset=sub_n * sub_warp_cols,
                            )

                return _Simplify(_gemm_ssr_scattered, inline_let=True)

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mfma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mfma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)

                if clear_accum:
                    T.clear(C_buf)

                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_buf, C_buf, ki)

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
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mfma ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
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

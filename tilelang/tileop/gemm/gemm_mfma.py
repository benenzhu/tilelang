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


class GemmMFMA(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MFMA)
        warp_row_tiles = int(self.M // m_warp) # 64
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
            scattered_warp=True,
            target=target,
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
            scattered_warp=True,
            target=target,
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

        # Determine if we should use pingpong (4-cluster) schedule
        pingpong = (mfma_emitter.scattered_warp and warp_rows >= 4 and warp_cols >= 4)
        sub_warp_rows = warp_rows // 2 if pingpong else warp_rows
        sub_warp_cols = warp_cols // 2 if pingpong else warp_cols

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Matrix Core mfma ops,
                accumulating into C_local.
                """
                num_ki = block_K // (micro_size_k * k_pack) if pingpong else 1
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)
                # Pingpong: B_local holds BOTH sub_n=0 and sub_n=1 simultaneously
                B_local_size = (warp_cols * local_size_b * k_pack * num_ki) if pingpong else (warp_cols * local_size_b * k_pack)
                B_local = T.alloc_local(B_local_size, in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                if pingpong:
                    # 4-cluster pingpong with merged ki, no reload:
                    #
                    # A_local[32]: sub_warp_rows(2) × local_size_a(8) × num_ki(2)
                    #   Holds one sub_m at a time, all ki values
                    #
                    # B_local[128]: sub_warp_cols(4) × local_size_b(8) × num_ki(2) × 2_sub_n
                    #   [0:64]   = sub_n=0, all ki
                    #   [64:128] = sub_n=1, all ki
                    #
                    # Cluster 0: ldA_sub0, ldB_sub0→[0:64],   mfma(0,0)
                    # Cluster 1:           ldB_sub1→[64:128], mfma(0,1) [reuse A]
                    # Cluster 2: ldA_sub1,                    mfma(1,0) [reuse B_sub0 from [0:64]]
                    # Cluster 3:                              mfma(1,1) [reuse A + B_sub1 from [64:128]]
                    sub_a_elems = sub_warp_rows * local_size_a * k_pack
                    sub_b_elems = sub_warp_cols * local_size_b * k_pack
                    b_sub1_base = sub_b_elems * num_ki  # = 64, start of sub_n=1 in B_local

                    # Cluster 0: load A sub_m=0, load B sub_n=0 → B_local[0:64]
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_a_subtile(A_local, A_region, ki, 0, sub_warp_rows,
                                                        local_offset=ki * sub_a_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_b_subtile(B_local, B_region, ki, 0, sub_warp_cols,
                                                        local_offset=ki * sub_b_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  0, sub_warp_rows, 0, sub_warp_cols, ki,
                                                  a_local_offset=ki * sub_a_elems,
                                                  b_local_offset=ki * sub_b_elems)

                    # Cluster 1: load B sub_n=1 → B_local[64:128], reuse A sub_m=0
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_b_subtile(B_local, B_region, ki, sub_warp_cols, sub_warp_cols,
                                                        local_offset=b_sub1_base + ki * sub_b_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  0, sub_warp_rows, sub_warp_cols, sub_warp_cols, ki,
                                                  a_local_offset=ki * sub_a_elems,
                                                  b_local_offset=b_sub1_base + ki * sub_b_elems)

                    # Cluster 2: load A sub_m=1, reuse B sub_n=0 from B_local[0:64]
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_a_subtile(A_local, A_region, ki, sub_warp_rows, sub_warp_rows,
                                                        local_offset=ki * sub_a_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  sub_warp_rows, sub_warp_rows, 0, sub_warp_cols, ki,
                                                  a_local_offset=ki * sub_a_elems,
                                                  b_local_offset=ki * sub_b_elems)

                    # Cluster 3: reuse A sub_m=1 + B sub_n=1 from B_local[64:128]
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  sub_warp_rows, sub_warp_rows, sub_warp_cols, sub_warp_cols, ki,
                                                  a_local_offset=ki * sub_a_elems,
                                                  b_local_offset=b_sub1_base + ki * sub_b_elems)
                else:
                    for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                        mfma_emitter.ldmatrix_a(A_local, A_region, ki)
                        mfma_emitter.ldmatrix_b(B_local, B_region, ki)
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

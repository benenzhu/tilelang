from __future__ import annotations

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

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
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
                # Pingpong: A_local holds BOTH sub_m=0 and sub_m=1 (64 elems)
                #           B_local holds one sub_n at a time (64 elems)
                # vs original A=32,B=64: +32 A, same B → same total
                # vs B=128 scheme: saves 64 B, costs 32 A → net -16 VGPR
                A_local_size = (warp_rows * local_size_a * k_pack * num_ki) if pingpong else (warp_rows * local_size_a * k_pack)
                A_local = T.alloc_local(A_local_size, in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                if pingpong:
                    # 4-cluster pingpong, swapped C1/C2, A double-buffered:
                    #
                    # A_local[64]: [0:32]=sub_m=0 all ki, [32:64]=sub_m=1 all ki
                    # B_local[64]: one sub_n at a time, overwritten between C1→C2
                    #
                    # C0: ldA_sub0→A[0:32],  ldB_sub0, mfma(0,0)
                    # C1: ldA_sub1→A[32:64],           mfma(1,0)  [B_sub0 reuse]
                    # C2: ldB_sub1 (overwrites B),     mfma(1,1)  [A_sub1 reuse]
                    # C3:                              mfma(0,1)  [A_sub0 + B_sub1 reuse]
                    sub_a_elems = sub_warp_rows * local_size_a * k_pack
                    sub_b_elems = sub_warp_cols * local_size_b * k_pack
                    a_sub1_base = sub_a_elems * num_ki  # offset to sub_m=1 in A_local

                    # Cluster 0: load A sub_m=0 → A[0:32], load B sub_n=0, mfma(0,0)
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

                    # Cluster 1: load A sub_m=1 → A[32:64], reuse B sub_n=0, mfma(1,0)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_a_subtile(A_local, A_region, ki, sub_warp_rows, sub_warp_rows,
                                                        local_offset=a_sub1_base + ki * sub_a_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  sub_warp_rows, sub_warp_rows, 0, sub_warp_cols, ki,
                                                  a_local_offset=a_sub1_base + ki * sub_a_elems,
                                                  b_local_offset=ki * sub_b_elems)

                    # Cluster 2: load B sub_n=1 (overwrites B), reuse A sub_m=1, mfma(1,1)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.ldmatrix_b_subtile(B_local, B_region, ki, sub_warp_cols, sub_warp_cols,
                                                        local_offset=ki * sub_b_elems)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  sub_warp_rows, sub_warp_rows, sub_warp_cols, sub_warp_cols, ki,
                                                  a_local_offset=a_sub1_base + ki * sub_a_elems,
                                                  b_local_offset=ki * sub_b_elems)

                    # Cluster 3: reuse A sub_m=0 from A[0:32] + B sub_n=1, mfma(0,1)
                    for ki in T.serial(0, num_ki):
                        mfma_emitter.mfma_subtile(A_local, B_local, C_buf,
                                                  0, sub_warp_rows, sub_warp_cols, sub_warp_cols, ki,
                                                  a_local_offset=ki * sub_a_elems,
                                                  b_local_offset=ki * sub_b_elems)
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

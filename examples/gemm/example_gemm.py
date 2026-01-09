# %%
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], verbose=True)
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@tilelang.jit(
    out_idx=[-1],
    verbose=True,
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "png",
    },
)
def matmul_nt(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        #### this bx, by is the usual cuda block idx. that the continues will be the bx;
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


kernel = None


def main(transpose_b=False):
    global kernel
    M, N, K = 19 * 256, 16 * 256, 8192
    if not transpose_b:
        kernel = matmul(M, N, K, 128, 128, 32)
    else:
        kernel = matmul_nt(M, N, K, 256, 256, 64)

    import torch

    a = torch.randn(M, K).cuda().bfloat16()

    if not transpose_b:
        b = torch.randn(K, N).cuda().bfloat16()
        c = kernel(a, b)
        ref_c = a @ b
    else:
        # NT layout: B is (N, K) contiguous
        b_nt = torch.randn(N, K).cuda().bfloat16()
        c = kernel(a, b_nt)
        ref_c = a @ b_nt.T

    # print("c:")
    # print(c)
    # print("ref_c:")
    # print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    # print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"flops: {2 * M * N * K / latency * 1e-9} Tflops")
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main(transpose_b=True)

/*
03_v0:
    use 512 thread
    simple load and calculate.
    pre register allocate.


*/




#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#ifndef PYTHON_CALL
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int SMEM_STRIDE = 64;
constexpr int NUM_WARP_M = BLOCK_M / 64;
constexpr int NUM_WARP_N = BLOCK_N / 64;
#endif

#define cdiv(a, b) (((a) + (b) - 1) / (b))


__launch_bounds__(512)
__global__
void _3_fp16_gemm_v0(
  const __hip_bfloat16 *A_gmem,
  const __hip_bfloat16 *B_gmem,
        __hip_bfloat16 *C_gmem
) {
    // __hip_bfloat16 Atile[]

    const int tid = threadIdx.x;

    // g2s
    // s2r
    // doing calculate
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    __shared__ __hip_bfloat16 A_smem[BLOCK_M * BLOCK_K];
    __shared__ __hip_bfloat16 B_smem[BLOCK_N * BLOCK_K];
    constexpr int total_shared_size_32768 = sizeof(A_smem) + sizeof(B_smem);
    constexpr int total_iter_num = cdiv(K, BLOCK_K);

    constexpr int C_reg_num__128 = BLOCK_M * BLOCK_N / 512;
    constexpr int warp_reg_num__8192 = C_reg_num__128 * 64;
    float C_reg0[C_reg_num__128/2];
    float C_reg1[C_reg_num__128/2];


    const int BM_idx = blockIdx.x / (cdiv(N, BLOCK_N)) * 256;
    const int BN_idx = blockIdx.x % (cdiv(N, BLOCK_N)) * 256;


    for(int iter = 0; iter < total_iter_num; iter++){
        { // g2s
            constexpr int total_need_transfer_num = total_shared_size_32768 / 512 / 4 / 4;
            //                                      total size                num   float  float4.
            for(int i = 0; i < total_need_transfer_num/2; i++){

                const int row_idx = (i * 16) / BLOCK_K;
                const int col_idx = (i * 16) % BLOCK_K;
                *(float4*)&A_smem[i] = *(float4*)&A_gmem[(BM_idx + row_idx) * K + (BN_idx + col_idx)];
            }

            for(int i = 0; i < total_need_transfer_num/2; i++){
                const int row_idx = (i * 16) / BLOCK_K;
                const int col_idx = (i * 16) % BLOCK_K;
                *(float4*)&B_smem[i] = *(float4*)&B_gmem[(BM_idx + row_idx) * K + (BN_idx + col_idx)];

            }
            __syncthreads();
        }

        { // s2r
            constexpr int AB_tile_reg__64 = (BLOCK_M / 4) * BLOCK_K / 64;
            __hip_bfloat16 B_reg[AB_tile_reg__64];
            __hip_bfloat16 A_reg0[AB_tile_reg__64];
            __hip_bfloat16 A_reg1[AB_tile_reg__64];
            // warp_M: 2 warp_N: 4
            const int warp_id_m = threadIdx.x / 64 / 4;
            const int warp_id_n = (threadIdx.x / 64) % 4;
            // B_reg: 64 * 64;
            for(int i = 0; i < 4; i++){ // large_row
                for(int j = 0; j < 4; j++){ // large_col
                    for(int k = 0; k < 4; k++){ // small_col
                        B_reg[i * 16 + j * 4 + k] = B_smem[
                            (warp_id_n * 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 16 + k + threadIdx.x / 16 * 4
                        ];
                    }
                }
            }
            for(int i = 0; i < 4; i++){ // large_row
                for(int j = 0; j < 4; j++){ // large_col
                    for(int k = 0; k < 4; k++){ // small_col
                        A_reg0[i * 16 + j * 4 + k] = B_smem[
                            (warp_id_m * 128 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 16 + k + threadIdx.x / 16 * 4
                        ];
                    }
                }
            }
            for(int i = 0; i < 4; i++){ // large_row
                for(int j = 0; j < 4; j++){ // large_col
                    for(int k = 0; k < 4; k++){ // small_col
                        A_reg1[i * 16 + j * 4 + k] = B_smem[
                            (warp_id_m * 128 + 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            (j * 16 + k + threadIdx.x / 16 * 4)
                        ];
                    }
                }
            }

            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    *((floatx4*)(C_reg0) + i * 4 + j)
                    = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                        *((float16x4*)(A_reg0) + i * 4 + j),
                        *((float16x4*)(B_reg) + i * 4 + j),
                        *((floatx4*)(C_reg0) + i * 4 + j),
                        0, 0, 0);
                }
            }
            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    *((floatx4*)(C_reg1) + i * 4 + j)
                    = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                        *((float16x4*)(A_reg1) + i * 4 + j),
                        *((float16x4*)(B_reg) + i * 4 + j),
                        *((floatx4*)(C_reg1) + i * 4 + j),
                        0, 0, 0);
                }
            }
            // mfma:

        } // s2r



    } // main loop for k_iter in ...

    // store C:
    for(int i = 0; i < 4; i++) { // row
        for(int j = 0; j < 4; j++) { // col
            for(int k = 0; k < 4; k++) { // small row
                C_gmem[ (BM_idx * 256 + i * 16 + k + threadIdx.x / 16 * 4) * N +
                        (BN_idx + j * 16 + threadIdx.x % 16)] = __float2bfloat16(
              C_reg0[i * 16 + j * 4 + k]);
            }
        }
    }


}

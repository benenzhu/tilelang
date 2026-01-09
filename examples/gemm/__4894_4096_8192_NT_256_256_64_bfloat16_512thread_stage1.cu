#include <hip/hip_runtime.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/debug.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(512)
    gemm_kernel(bfloat16_t *__restrict__ A, bfloat16_t *__restrict__ B,
                bfloat16_t *__restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  bfloat16_t A_local[16];
  bfloat16_t B_local[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float4 *)(C_local + (i * 4)) =
        make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
#pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(
        (&(((bfloat16_t *)buf_dyn_shmem)
               [i_1 * 4096 + ((int)threadIdx.x >> 3) * 64 +
                ((((int)threadIdx.x & 63) >> 5) +
                     (((int)threadIdx.x & 7) >> 2) &
                 1) *
                    32 +
                ((((int)threadIdx.x & 31) >> 4) +
                     (((int)threadIdx.x & 3) >> 1) &
                 1) *
                    16 +
                ((((int)threadIdx.x & 15) >> 3) + ((int)threadIdx.x & 1) & 1) *
                    8])),
        (&(A[(int)blockIdx.y * 2097152 + i_1 * 524288 +
             ((int)threadIdx.x >> 3) * 8192 + ((int)threadIdx.x & 7) * 8])));
  }
#pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(
        (&(((bfloat16_t *)buf_dyn_shmem)
               [i_2 * 4096 + ((int)threadIdx.x >> 3) * 64 +
                ((((int)threadIdx.x & 63) >> 5) +
                     (((int)threadIdx.x & 7) >> 2) &
                 1) *
                    32 +
                ((((int)threadIdx.x & 31) >> 4) +
                     (((int)threadIdx.x & 3) >> 1) &
                 1) *
                    16 +
                ((((int)threadIdx.x & 15) >> 3) + ((int)threadIdx.x & 1) & 1) *
                    8 +
                16384])),
        (&(B[(int)blockIdx.x * 2097152 + i_2 * 524288 +
             ((int)threadIdx.x >> 3) * 8192 + ((int)threadIdx.x & 7) * 8])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 127; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        *(uint2 *)(A_local + (i_3 * 4)) = *(
            uint2 *)(((bfloat16_t *)buf_dyn_shmem) +
                     ((((int)threadIdx.x & 255) >> 6) * 4096 + i_3 * 1024 +
                      ((int)threadIdx.x & 15) * 64 +
                      ((((int)threadIdx.x & 7) >> 2) + (ki >> 1) & 1) * 32 +
                      ((((int)threadIdx.x & 3) >> 1) + (ki & 1) & 1) * 16 +
                      ((((int)threadIdx.x & 63) >> 5) + ((int)threadIdx.x & 1) &
                       1) *
                          8 +
                      (((int)threadIdx.x & 31) >> 4) * 4));
      }
      for (int j = 0; j < 8; ++j) {
        *(uint2 *)(B_local + (j * 4)) = *(
            uint2 *)(((bfloat16_t *)buf_dyn_shmem) +
                     (((int)threadIdx.x >> 8) * 8192 + j * 1024 +
                      ((int)threadIdx.x & 15) * 64 +
                      ((((int)threadIdx.x & 7) >> 2) + (ki >> 1) & 1) * 32 +
                      ((((int)threadIdx.x & 3) >> 1) + (ki & 1) & 1) * 16 +
                      ((((int)threadIdx.x & 63) >> 5) + ((int)threadIdx.x & 1) &
                       1) *
                          8 +
                      (((int)threadIdx.x & 31) >> 4) * 4 + 16384));
      }
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
            *(((float32x4 *)C_local) + ((i_4 * 8) + j_1)) =
                __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    *(((bfloat16x4_vec *)B_local) + j_1),
                    *(((bfloat16x4_vec *)A_local) + i_4),
                    *(((float32x4 *)C_local) + ((i_4 * 8) + j_1)), 0, 0, 0);
          };
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>(
          (&(((bfloat16_t *)
                  buf_dyn_shmem)[i_5 * 4096 + ((int)threadIdx.x >> 3) * 64 +
                                 ((((int)threadIdx.x & 63) >> 5) +
                                      (((int)threadIdx.x & 7) >> 2) &
                                  1) *
                                     32 +
                                 ((((int)threadIdx.x & 31) >> 4) +
                                      (((int)threadIdx.x & 3) >> 1) &
                                  1) *
                                     16 +
                                 ((((int)threadIdx.x & 15) >> 3) +
                                      ((int)threadIdx.x & 1) &
                                  1) *
                                     8])),
          (&(A[(int)blockIdx.y * 2097152 + i_5 * 524288 +
               ((int)threadIdx.x >> 3) * 8192 + k * 64 +
               ((int)threadIdx.x & 7) * 8 + 64])));
    }
#pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>(
          (&(((bfloat16_t *)
                  buf_dyn_shmem)[i_6 * 4096 + ((int)threadIdx.x >> 3) * 64 +
                                 ((((int)threadIdx.x & 63) >> 5) +
                                      (((int)threadIdx.x & 7) >> 2) &
                                  1) *
                                     32 +
                                 ((((int)threadIdx.x & 31) >> 4) +
                                      (((int)threadIdx.x & 3) >> 1) &
                                  1) *
                                     16 +
                                 ((((int)threadIdx.x & 15) >> 3) +
                                      ((int)threadIdx.x & 1) &
                                  1) *
                                     8 +
                                 16384])),
          (&(B[(int)blockIdx.x * 2097152 + i_6 * 524288 +
               ((int)threadIdx.x >> 3) * 8192 + k * 64 +
               ((int)threadIdx.x & 7) * 8 + 64])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint2 *)(A_local + (i_7 * 4)) =
          *(uint2 *)(((bfloat16_t *)buf_dyn_shmem) +
                     ((((int)threadIdx.x & 255) >> 6) * 4096 + i_7 * 1024 +
                      ((int)threadIdx.x & 15) * 64 +
                      ((((int)threadIdx.x & 7) >> 2) + (ki_1 >> 1) & 1) * 32 +
                      ((((int)threadIdx.x & 3) >> 1) + (ki_1 & 1) & 1) * 16 +
                      ((((int)threadIdx.x & 63) >> 5) + ((int)threadIdx.x & 1) &
                       1) *
                          8 +
                      (((int)threadIdx.x & 31) >> 4) * 4));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      *(uint2 *)(B_local + (j_2 * 4)) =
          *(uint2 *)(((bfloat16_t *)buf_dyn_shmem) +
                     (((int)threadIdx.x >> 8) * 8192 + j_2 * 1024 +
                      ((int)threadIdx.x & 15) * 64 +
                      ((((int)threadIdx.x & 7) >> 2) + (ki_1 >> 1) & 1) * 32 +
                      ((((int)threadIdx.x & 3) >> 1) + (ki_1 & 1) & 1) * 16 +
                      ((((int)threadIdx.x & 63) >> 5) + ((int)threadIdx.x & 1) &
                       1) *
                          8 +
                      (((int)threadIdx.x & 31) >> 4) * 4 + 16384));
    }
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
          *(((float32x4 *)C_local) + ((i_8 * 8) + j_3)) =
              __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                  *(((bfloat16x4_vec *)B_local) + j_3),
                  *(((bfloat16x4_vec *)A_local) + i_8),
                  *(((float32x4 *)C_local) + ((i_8 * 8) + j_3)), 0, 0, 0);
        };
      }
    }
  }
#pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    uint2 __1;
    float4 v_ = *(float4 *)(C_local + (i_9 * 4));
    ((bfloat16_t *)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t *)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t *)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t *)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
    *(uint2 *)(C + ((int)blockIdx.y * 1048576 +
                    (((int)threadIdx.x & 255) >> 6) * 262144 +
                    (i_9 >> 3) * 65536 + ((int)threadIdx.x & 15) * 4096 +
                    (int)blockIdx.x * 256 + ((int)threadIdx.x >> 8) * 128 +
                    (i_9 & 7) * 16 + (((int)threadIdx.x & 63) >> 4) * 4)) = __1;
  }
}

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char *get_last_error() { return error_buf; }

extern "C" int init() {
  error_buf[0] = '\0';

  if (65536 > 65536) {
    snprintf(error_buf, ERROR_BUF_SIZE,
             "Failed to set the allowed dynamic shared memory size for "
             "gemm_kernel to %d",
             65536);
    return -1;
  }
  return 0;

  return 0;
}

extern "C" int call(bfloat16_t *__restrict__ A, bfloat16_t *__restrict__ B,
                    bfloat16_t *__restrict__ C,
                    hipStream_t stream = hipStreamDefault) {
  gemm_kernel<<<dim3(16, 19, 1), dim3(512, 1, 1), 65536, stream>>>(A, B, C);
  TILELANG_CHECK_LAST_ERROR("gemm_kernel");

  return 0;
}

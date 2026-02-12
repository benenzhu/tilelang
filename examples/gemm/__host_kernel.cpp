#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(512) gemm_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  float C_local[128];
  __shared__ __align__(1024) bfloat16_t A_shared[32768];
  __shared__ __align__(1024) bfloat16_t B_shared[32768];
  bfloat16_t A_local[64];
  bfloat16_t B_local[16];
  bfloat16_t C_local_cast[4];
  const dim3 blockIdx = tl::rasterization2DRowXcd<4, 8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(A_shared[((i_1 * 4096) + (((int)threadIdx.x) * 8))])), (&(A[((((((((int)blockIdx.y) * 2097152) + (i_1 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])));
  }
  __builtin_amdgcn_s_barrier();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(B_shared[((i_2 * 4096) + (((int)threadIdx.x) * 8))])), (&(B[((((((((int)blockIdx.x) * 2097152) + (i_2 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 127; ++k) {
    __builtin_amdgcn_s_barrier();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(A_shared[(((((k + 1) & 1) * 16384) + (i_3 * 4096)) + (((int)threadIdx.x) * 8))])), (&(A[((((((((((int)blockIdx.y) * 2097152) + (i_3 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (k * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)])));
    }
    // __builtin_amdgcn_s_barrier();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(B_shared[(((((k + 1) & 1) * 16384) + (i_4 * 4096)) + (((int)threadIdx.x) * 8))])), (&(B[((((((((((int)blockIdx.x) * 2097152) + (i_4 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (k * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<8>();
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id = 0; local_id < 2; ++local_id) {
      *(uint4*)(A_local + (local_id * 8)) = *(uint4*)(A_shared + (((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
    for (int local_id_1 = 0; local_id_1 < 2; ++local_id_1) {
      *(uint4*)(A_local + ((local_id_1 * 8) + 16)) = *(uint4*)(A_shared + ((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_1 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1024));
    }
    for (int local_id_2 = 0; local_id_2 < 2; ++local_id_2) {
      *(uint4*)(B_local + (local_id_2 * 8)) = *(uint4*)(B_shared + (((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_2 + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 1" : "+v"(*(((float32x4*)C_local) + 0)), "+v"(*(((float32x4*)C_local) + 8))   : : "memory");
    for (int kp = 0; kp < 2; ++kp) {
      {
      *(((float32x4*)C_local) + 0) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp),
                    *(((bfloat16x8_vec*)A_local) + kp),
                    *(((float32x4*)C_local) + 0), 0, 0, 0);
    };
    }
    for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
      {
      *(((float32x4*)C_local) + 8) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_1),
                    *(((bfloat16x8_vec*)A_local) + (kp_1 + 2)),
                    *(((float32x4*)C_local) + 8), 0, 0, 0);
    };
    }
    asm volatile("s_setprio 0" : : "v"(*(((float32x4*)C_local) + 0)), "v"(*(((float32x4*)C_local) + 8)) : "memory");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_3 = 0; local_id_3 < 2; ++local_id_3) {
      *(uint4*)(A_local + ((local_id_3 * 8) + 32)) = *(uint4*)(A_shared + ((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_3 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_4 = 0; local_id_4 < 2; ++local_id_4) {
      *(uint4*)(A_local + ((local_id_4 * 8) + 48)) = *(uint4*)(A_shared + ((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_4 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 3072));
    }
    __builtin_amdgcn_sched_barrier(0);

    asm volatile("s_setprio 1" : "+v"(*(((float32x4*)C_local) + 16)), "+v"(*(((float32x4*)C_local) + 24)) : : "memory");
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_2 = 0; kp_2 < 2; ++kp_2) {
      {
      *(((float32x4*)C_local) + 16) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_2),
                    *(((bfloat16x8_vec*)A_local) + (kp_2 + 4)),
                    *(((float32x4*)C_local) + 16), 0, 0, 0);
    };
    }
    for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
      {
      *(((float32x4*)C_local) + 24) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_3),
                    *(((bfloat16x8_vec*)A_local) + (kp_3 + 6)),
                    *(((float32x4*)C_local) + 24), 0, 0, 0);
    };
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 0" : : "v"(*(((float32x4*)C_local) + 16)), "v"(*(((float32x4*)C_local) + 24)) : "memory");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_5 = 0; local_id_5 < 2; ++local_id_5) {
      *(uint4*)(B_local + (local_id_5 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_5 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1024));
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    for (int kp_4 = 0; kp_4 < 2; ++kp_4) {
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        {
      *(((float32x4*)C_local) + ((i_5 * 8) + 1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_4),
                    *(((bfloat16x8_vec*)A_local) + ((i_5 * 2) + kp_4)),
                    *(((float32x4*)C_local) + ((i_5 * 8) + 1)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_6 = 0; local_id_6 < 2; ++local_id_6) {
      *(uint4*)(B_local + (local_id_6 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_6 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_5 = 0; kp_5 < 2; ++kp_5) {
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        {
      *(((float32x4*)C_local) + ((i_6 * 8) + 2)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_5),
                    *(((bfloat16x8_vec*)A_local) + ((i_6 * 2) + kp_5)),
                    *(((float32x4*)C_local) + ((i_6 * 8) + 2)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_7 = 0; local_id_7 < 2; ++local_id_7) {
      *(uint4*)(B_local + (local_id_7 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_7 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 3072));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_6 = 0; kp_6 < 2; ++kp_6) {
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        {
      *(((float32x4*)C_local) + ((i_7 * 8) + 3)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_6),
                    *(((bfloat16x8_vec*)A_local) + ((i_7 * 2) + kp_6)),
                    *(((float32x4*)C_local) + ((i_7 * 8) + 3)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_8 = 0; local_id_8 < 2; ++local_id_8) {
      *(uint4*)(B_local + (local_id_8 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_8 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_7 = 0; kp_7 < 2; ++kp_7) {
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        {
      *(((float32x4*)C_local) + ((i_8 * 8) + 4)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_7),
                    *(((bfloat16x8_vec*)A_local) + ((i_8 * 2) + kp_7)),
                    *(((float32x4*)C_local) + ((i_8 * 8) + 4)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_9 = 0; local_id_9 < 2; ++local_id_9) {
      *(uint4*)(B_local + (local_id_9 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_9 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 5120));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_8 = 0; kp_8 < 2; ++kp_8) {
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        {
      *(((float32x4*)C_local) + ((i_9 * 8) + 5)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_8),
                    *(((bfloat16x8_vec*)A_local) + ((i_9 * 2) + kp_8)),
                    *(((float32x4*)C_local) + ((i_9 * 8) + 5)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_10 = 0; local_id_10 < 2; ++local_id_10) {
      *(uint4*)(B_local + (local_id_10 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_10 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6144));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_9 = 0; kp_9 < 2; ++kp_9) {
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        {
      *(((float32x4*)C_local) + ((i_10 * 8) + 6)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_9),
                    *(((bfloat16x8_vec*)A_local) + ((i_10 * 2) + kp_9)),
                    *(((float32x4*)C_local) + ((i_10 * 8) + 6)), 0, 0, 0);
    };
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int local_id_11 = 0; local_id_11 < 2; ++local_id_11) {
      *(uint4*)(B_local + (local_id_11 * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_11 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7168));
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int kp_10 = 0; kp_10 < 2; ++kp_10) {
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        {
      *(((float32x4*)C_local) + ((i_11 * 8) + 7)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_10),
                    *(((bfloat16x8_vec*)A_local) + ((i_11 * 2) + kp_10)),
                    *(((float32x4*)C_local) + ((i_11 * 8) + 7)), 0, 0, 0);
    };
      }
    }
  }
  tl::cp_async_wait<0>();
  __builtin_amdgcn_s_barrier();
  for (int local_id_12 = 0; local_id_12 < 2; ++local_id_12) {
    *(uint4*)(A_local + (local_id_12 * 8)) = *(uint4*)(A_shared + ((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_12 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
  }
  for (int local_id_13 = 0; local_id_13 < 2; ++local_id_13) {
    *(uint4*)(A_local + ((local_id_13 * 8) + 16)) = *(uint4*)(A_shared + ((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_13 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 17408));
  }
  for (int local_id_14 = 0; local_id_14 < 2; ++local_id_14) {
    *(uint4*)(B_local + (local_id_14 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_14 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
  }
  for (int kp_11 = 0; kp_11 < 2; ++kp_11) {
    {
      *(((float32x4*)C_local) + 0) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_11),
                    *(((bfloat16x8_vec*)A_local) + kp_11),
                    *(((float32x4*)C_local) + 0), 0, 0, 0);
    };
  }
  for (int kp_12 = 0; kp_12 < 2; ++kp_12) {
    {
      *(((float32x4*)C_local) + 8) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_12),
                    *(((bfloat16x8_vec*)A_local) + (kp_12 + 2)),
                    *(((float32x4*)C_local) + 8), 0, 0, 0);
    };
  }
  for (int local_id_15 = 0; local_id_15 < 2; ++local_id_15) {
    *(uint4*)(A_local + ((local_id_15 * 8) + 32)) = *(uint4*)(A_shared + ((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_15 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18432));
  }
  for (int local_id_16 = 0; local_id_16 < 2; ++local_id_16) {
    *(uint4*)(A_local + ((local_id_16 * 8) + 48)) = *(uint4*)(A_shared + ((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_16 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 19456));
  }
  for (int kp_13 = 0; kp_13 < 2; ++kp_13) {
    {
      *(((float32x4*)C_local) + 16) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_13),
                    *(((bfloat16x8_vec*)A_local) + (kp_13 + 4)),
                    *(((float32x4*)C_local) + 16), 0, 0, 0);
    };
  }
  for (int kp_14 = 0; kp_14 < 2; ++kp_14) {
    {
      *(((float32x4*)C_local) + 24) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_14),
                    *(((bfloat16x8_vec*)A_local) + (kp_14 + 6)),
                    *(((float32x4*)C_local) + 24), 0, 0, 0);
    };
  }
  for (int local_id_17 = 0; local_id_17 < 2; ++local_id_17) {
    *(uint4*)(B_local + (local_id_17 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_17 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 17408));
  }
  for (int kp_15 = 0; kp_15 < 2; ++kp_15) {
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      {
      *(((float32x4*)C_local) + ((i_12 * 8) + 1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_15),
                    *(((bfloat16x8_vec*)A_local) + ((i_12 * 2) + kp_15)),
                    *(((float32x4*)C_local) + ((i_12 * 8) + 1)), 0, 0, 0);
    };
    }
  }
  for (int local_id_18 = 0; local_id_18 < 2; ++local_id_18) {
    *(uint4*)(B_local + (local_id_18 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_18 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18432));
  }
  for (int kp_16 = 0; kp_16 < 2; ++kp_16) {
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      {
      *(((float32x4*)C_local) + ((i_13 * 8) + 2)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_16),
                    *(((bfloat16x8_vec*)A_local) + ((i_13 * 2) + kp_16)),
                    *(((float32x4*)C_local) + ((i_13 * 8) + 2)), 0, 0, 0);
    };
    }
  }
  for (int local_id_19 = 0; local_id_19 < 2; ++local_id_19) {
    *(uint4*)(B_local + (local_id_19 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_19 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 19456));
  }
  for (int kp_17 = 0; kp_17 < 2; ++kp_17) {
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      {
      *(((float32x4*)C_local) + ((i_14 * 8) + 3)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_17),
                    *(((bfloat16x8_vec*)A_local) + ((i_14 * 2) + kp_17)),
                    *(((float32x4*)C_local) + ((i_14 * 8) + 3)), 0, 0, 0);
    };
    }
  }
  for (int local_id_20 = 0; local_id_20 < 2; ++local_id_20) {
    *(uint4*)(B_local + (local_id_20 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_20 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20480));
  }
  for (int kp_18 = 0; kp_18 < 2; ++kp_18) {
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      {
      *(((float32x4*)C_local) + ((i_15 * 8) + 4)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_18),
                    *(((bfloat16x8_vec*)A_local) + ((i_15 * 2) + kp_18)),
                    *(((float32x4*)C_local) + ((i_15 * 8) + 4)), 0, 0, 0);
    };
    }
  }
  for (int local_id_21 = 0; local_id_21 < 2; ++local_id_21) {
    *(uint4*)(B_local + (local_id_21 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_21 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 21504));
  }
  for (int kp_19 = 0; kp_19 < 2; ++kp_19) {
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      {
      *(((float32x4*)C_local) + ((i_16 * 8) + 5)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_19),
                    *(((bfloat16x8_vec*)A_local) + ((i_16 * 2) + kp_19)),
                    *(((float32x4*)C_local) + ((i_16 * 8) + 5)), 0, 0, 0);
    };
    }
  }
  for (int local_id_22 = 0; local_id_22 < 2; ++local_id_22) {
    *(uint4*)(B_local + (local_id_22 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_22 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22528));
  }
  for (int kp_20 = 0; kp_20 < 2; ++kp_20) {
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      {
      *(((float32x4*)C_local) + ((i_17 * 8) + 6)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_20),
                    *(((bfloat16x8_vec*)A_local) + ((i_17 * 2) + kp_20)),
                    *(((float32x4*)C_local) + ((i_17 * 8) + 6)), 0, 0, 0);
    };
    }
  }
  for (int local_id_23 = 0; local_id_23 < 2; ++local_id_23) {
    *(uint4*)(B_local + (local_id_23 * 8)) = *(uint4*)(B_shared + (((((((((int)threadIdx.x) >> 8) * 8192) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_23 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 23552));
  }
  for (int kp_21 = 0; kp_21 < 2; ++kp_21) {
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      {
      *(((float32x4*)C_local) + ((i_18 * 8) + 7)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + kp_21),
                    *(((bfloat16x8_vec*)A_local) + ((i_18 * 2) + kp_21)),
                    *(((float32x4*)C_local) + ((i_18 * 8) + 7)), 0, 0, 0);
    };
    }
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 32; ++i_19) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_19 * 4));
    ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + ((((((((((int)blockIdx.y) * 2097152) + (((((int)threadIdx.x) & 255) >> 6) * 524288)) + ((i_19 >> 3) * 131072)) + ((((int)threadIdx.x) & 15) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 8) * 128)) + ((i_19 & 7) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    int device_gemm_kernel = 0;
    hipError_t dev_res_gemm_kernel = hipGetDevice(&device_gemm_kernel);
    if (dev_res_gemm_kernel != hipSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to get HIP device for gemm_kernel: %s", hipGetErrorString(dev_res_gemm_kernel));
        return -1;
    }
    int max_smem_gemm_kernel = 0;
    hipError_t attr_res_gemm_kernel = hipDeviceGetAttribute(&max_smem_gemm_kernel, hipDeviceAttributeMaxSharedMemoryPerBlock, device_gemm_kernel);
    if (attr_res_gemm_kernel != hipSuccess || max_smem_gemm_kernel <= 0) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to query HIP max shared memory for gemm_kernel: %s", hipGetErrorString(attr_res_gemm_kernel));
        return -1;
    }
    if (131072 > max_smem_gemm_kernel) {
        snprintf(
            error_buf,
            ERROR_BUF_SIZE,
            "Requested dynamic shared memory %d exceeds device limit %d for gemm_kernel",
            131072,
            max_smem_gemm_kernel
        );
        return -1;
    }
    return 0;

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C, hipStream_t stream=hipStreamDefault) {
	gemm_kernel<<<dim3(32, 32, 1), dim3(512, 1, 1), 0, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("gemm_kernel");

	return 0;
}

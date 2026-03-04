#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(512) gemm_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  auto __rsrc_A = make_wave_buffer_resource((const void*)(A));
  uint32_t __base_A = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(A));
  auto __rsrc_B = make_wave_buffer_resource((const void*)(B));
  uint32_t __base_B = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(B));
  float C_local[128];
  __shared__ __align__(1024) bfloat16_t A_shared[32768];
  __shared__ __align__(1024) bfloat16_t B_shared[32768];
  bfloat16_t A_local[32];
  bfloat16_t B_local[64];
  bfloat16_t C_local_cast[4];
  int __g2s_thread_offset;
  asm volatile("v_mov_b32 %0, %1" : "=v"(__g2s_thread_offset) : "v"((((((threadIdx.x >> 3) * 8192) + (((((threadIdx.x & 63) >> 5) + ((threadIdx.x & 7) >> 2)) & 1) * 32)) + (((((threadIdx.x & 31) >> 4) + ((threadIdx.x & 3) >> 1)) & 1) * 16)) + (((((threadIdx.x & 15) >> 3) + (threadIdx.x & 1)) & 1) * 8))));
  const dim3 blockIdx = tl::rasterization2DRow<4, true>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + i * 4) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_lds_voffset<16>((&(A_shared[i_1 * 4096 + threadIdx.x * 8])), (uint32_t)(((((blockIdx.y * 2097152) + (i_1 * 524288)) + __g2s_thread_offset) * 2)), __rsrc_A);
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs_lds_voffset<16>((&(B_shared[i_2 * 4096 + threadIdx.x * 8])), (uint32_t)(((((blockIdx.x * 2097152) + (i_2 * 524288)) + __g2s_thread_offset) * 2)), __rsrc_B);
  }
  tl::cp_async_commit();
  for (int k = 0; k < 127; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs_lds_voffset<16>((&(A_shared[(k + 1 & 1) * 16384 + i_3 * 4096 + threadIdx.x * 8])), (uint32_t)(((((((blockIdx.y * 2097152) + (i_3 * 524288)) + (k * 64)) + __g2s_thread_offset) + 64) * 2)), __rsrc_A);
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs_lds_voffset<16>((&(B_shared[(k + 1 & 1) * 16384 + i_4 * 4096 + threadIdx.x * 8])), (uint32_t)(((((((blockIdx.x * 2097152) + (i_4 * 524288)) + (k * 64)) + __g2s_thread_offset) + 64) * 2)), __rsrc_B);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<8>();
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        *(uint4*)(A_local + i_5 * 8) = *(uint4*)(A_shared + ((k & 1) * 16384 + ((threadIdx.x & 255) >> 6) * 2048 + i_5 * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8));
      }
      for (int j = 0; j < 4; ++j) {
        *(uint4*)(B_local + j * 8) = *(uint4*)(B_shared + ((k & 1) * 16384 + (threadIdx.x >> 8) * 4096 + j * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8));
      }
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          {
      *(((float32x4*)C_local) + ((i_6 * 8) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_1),
                    *(((bfloat16x8_vec*)A_local) + i_6),
                    *(((float32x4*)C_local) + ((i_6 * 8) + j_1)), 0, 0, 0);
    };
        }
      }
      for (int jj = 0; jj < 4; ++jj) {
        *(uint4*)(B_local + (jj * 8 + 32)) = *(uint4*)(B_shared + ((k & 1) * 16384 + (threadIdx.x >> 8) * 4096 + jj * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 8192));
      }
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        for (int jj_1 = 0; jj_1 < 4; ++jj_1) {
          {
      *(((float32x4*)C_local) + (((i_7 * 8) + jj_1) + 4)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + (jj_1 + 4)),
                    *(((bfloat16x8_vec*)A_local) + i_7),
                    *(((float32x4*)C_local) + (((i_7 * 8) + jj_1) + 4)), 0, 0, 0);
    };
        }
      }
      for (int ii = 0; ii < 2; ++ii) {
        *(uint4*)(A_local + (ii * 8 + 16)) = *(uint4*)(A_shared + ((k & 1) * 16384 + ((threadIdx.x & 255) >> 6) * 2048 + ii * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 8192));
      }
      for (int ii_1 = 0; ii_1 < 2; ++ii_1) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          {
      *(((float32x4*)C_local) + (((ii_1 * 8) + j_2) + 16)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_2),
                    *(((bfloat16x8_vec*)A_local) + (ii_1 + 2)),
                    *(((float32x4*)C_local) + (((ii_1 * 8) + j_2) + 16)), 0, 0, 0);
    };
        }
      }
      for (int ii_2 = 0; ii_2 < 2; ++ii_2) {
        for (int jj_2 = 0; jj_2 < 4; ++jj_2) {
          {
      *(((float32x4*)C_local) + (((ii_2 * 8) + jj_2) + 20)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + (jj_2 + 4)),
                    *(((bfloat16x8_vec*)A_local) + (ii_2 + 2)),
                    *(((float32x4*)C_local) + (((ii_2 * 8) + jj_2) + 20)), 0, 0, 0);
    };
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      *(uint4*)(A_local + i_8 * 8) = *(uint4*)(A_shared + (((threadIdx.x & 255) >> 6) * 2048 + i_8 * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki_1 & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 16384));
    }
    for (int j_3 = 0; j_3 < 4; ++j_3) {
      *(uint4*)(B_local + j_3 * 8) = *(uint4*)(B_shared + ((threadIdx.x >> 8) * 4096 + j_3 * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki_1 & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 16384));
    }
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      for (int j_4 = 0; j_4 < 4; ++j_4) {
        {
      *(((float32x4*)C_local) + ((i_9 * 8) + j_4)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_4),
                    *(((bfloat16x8_vec*)A_local) + i_9),
                    *(((float32x4*)C_local) + ((i_9 * 8) + j_4)), 0, 0, 0);
    };
      }
    }
    for (int jj_3 = 0; jj_3 < 4; ++jj_3) {
      *(uint4*)(B_local + (jj_3 * 8 + 32)) = *(uint4*)(B_shared + ((threadIdx.x >> 8) * 4096 + jj_3 * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki_1 & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 24576));
    }
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      for (int jj_4 = 0; jj_4 < 4; ++jj_4) {
        {
      *(((float32x4*)C_local) + (((i_10 * 8) + jj_4) + 4)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + (jj_4 + 4)),
                    *(((bfloat16x8_vec*)A_local) + i_10),
                    *(((float32x4*)C_local) + (((i_10 * 8) + jj_4) + 4)), 0, 0, 0);
    };
      }
    }
    for (int ii_3 = 0; ii_3 < 2; ++ii_3) {
      *(uint4*)(A_local + (ii_3 * 8 + 16)) = *(uint4*)(A_shared + (((threadIdx.x & 255) >> 6) * 2048 + ii_3 * 1024 + (threadIdx.x & 15) * 64 + (((threadIdx.x & 7) >> 2) + ki_1 & 1) * 32 + (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 3) >> 1) & 1) * 16 + (((threadIdx.x & 31) >> 4) + (threadIdx.x & 1) & 1) * 8 + 24576));
    }
    for (int ii_4 = 0; ii_4 < 2; ++ii_4) {
      for (int j_5 = 0; j_5 < 4; ++j_5) {
        {
      *(((float32x4*)C_local) + (((ii_4 * 8) + j_5) + 16)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_5),
                    *(((bfloat16x8_vec*)A_local) + (ii_4 + 2)),
                    *(((float32x4*)C_local) + (((ii_4 * 8) + j_5) + 16)), 0, 0, 0);
    };
      }
    }
    for (int ii_5 = 0; ii_5 < 2; ++ii_5) {
      for (int jj_5 = 0; jj_5 < 4; ++jj_5) {
        {
      *(((float32x4*)C_local) + (((ii_5 * 8) + jj_5) + 20)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + (jj_5 + 4)),
                    *(((bfloat16x8_vec*)A_local) + (ii_5 + 2)),
                    *(((float32x4*)C_local) + (((ii_5 * 8) + jj_5) + 20)), 0, 0, 0);
    };
      }
    }
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 32; ++i_11) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + i_11 * 4);
    ((bfloat16_t*)(&__1.x))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t*)(&__1.x))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t*)(&__1.y))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t*)(&__1.y))[1] = (bfloat16_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + (blockIdx.y * 2097152 + (i_11 >> 4) * 1048576 + ((threadIdx.x & 255) >> 6) * 262144 + ((i_11 & 15) >> 3) * 131072 + (threadIdx.x & 15) * 8192 + blockIdx.x * 256 + ((i_11 & 7) >> 2) * 128 + (threadIdx.x >> 8) * 64 + (i_11 & 3) * 16 + ((threadIdx.x & 63) >> 4) * 4)) = *(uint2*)(C_local_cast + 0);
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C, hipStream_t stream=hipStreamDefault) {
	gemm_kernel<<<dim3(32, 32, 1), dim3(512, 1, 1), 0, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("gemm_kernel");

	return 0;
}

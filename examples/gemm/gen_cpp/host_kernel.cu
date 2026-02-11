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
  bfloat16_t B_local[128];
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
    __builtin_amdgcn_s_barrier();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(B_shared[(((((k + 1) & 1) * 16384) + (i_4 * 4096)) + (((int)threadIdx.x) * 8))])), (&(B[((((((((((int)blockIdx.x) * 2097152) + (i_4 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (k * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<8>();
    __builtin_amdgcn_s_barrier();
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      for (int local_id = 0; local_id < 2; ++local_id) {
        *(uint4*)(A_local + ((i_5 * 16) + (local_id * 8))) = *(uint4*)(A_shared + ((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + (i_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id + (((int)threadIdx.x) & 1)) & 1) * 8)));
      }
    }
    for (int j = 0; j < 8; ++j) {
      for (int local_id_1 = 0; local_id_1 < 2; ++local_id_1) {
        *(uint4*)(B_local + ((j * 16) + (local_id_1 * 8))) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_1 + (((int)threadIdx.x) & 1)) & 1) * 8)));
      }
    }
    for (int kp = 0; kp < 2; ++kp) {
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)C_local) + ((i_6 * 8) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + ((j_1 * 2) + kp)),
                    *(((bfloat16x8_vec*)A_local) + ((i_6 * 2) + kp)),
                    *(((float32x4*)C_local) + ((i_6 * 8) + j_1)), 0, 0, 0);
    };
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __builtin_amdgcn_s_barrier();
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    for (int local_id_2 = 0; local_id_2 < 2; ++local_id_2) {
      *(uint4*)(A_local + ((i_7 * 16) + (local_id_2 * 8))) = *(uint4*)(A_shared + (((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_2 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
    }
  }
  for (int j_2 = 0; j_2 < 8; ++j_2) {
    for (int local_id_3 = 0; local_id_3 < 2; ++local_id_3) {
      *(uint4*)(B_local + ((j_2 * 16) + (local_id_3 * 8))) = *(uint4*)(B_shared + ((((((((((int)threadIdx.x) >> 8) * 8192) + (j_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((local_id_3 + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
    }
  }
  for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)C_local) + ((i_8 * 8) + j_3)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + ((j_3 * 2) + kp_1)),
                    *(((bfloat16x8_vec*)A_local) + ((i_8 * 2) + kp_1)),
                    *(((float32x4*)C_local) + ((i_8 * 8) + j_3)), 0, 0, 0);
    };
      }
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_9 * 4));
    ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + ((((((((((int)blockIdx.y) * 2097152) + (((((int)threadIdx.x) & 255) >> 6) * 524288)) + ((i_9 >> 3) * 131072)) + ((((int)threadIdx.x) & 15) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 8) * 128)) + ((i_9 & 7) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(C_local_cast + 0);
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

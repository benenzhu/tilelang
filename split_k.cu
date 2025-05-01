#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  if (((int)blockIdx.z) == 0) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      *(uint4*)(C + (((((((int)blockIdx.y) * 262144) + (i * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = make_uint4(__pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)));
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    *(float2*)(C_local + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  for (int ko = 0; ko < 8; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = *(uint4*)(A + ((((((((int)blockIdx.y) * 131072) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (((int)blockIdx.z) * 256)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      uint4 condval;
      if ((((int)blockIdx.x) < 8)) {
        condval = *(uint4*)(B + ((((((((int)blockIdx.z) * 262144) + (ko * 32768)) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
      } else {
        condval = make_uint4(__pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)));
      }
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
    }
    __syncthreads();
    tl::gemm_ss<128, 128, 32, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[0])), (&(C_local[0])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_4 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((i_4 & 7) >> 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_4 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_4 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 64; ++i_5) {
    AtomicAddx2((&(C[(((((((int)blockIdx.y) * 262144) + (i_5 * 4096)) + ((((int)threadIdx.x) >> 6) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 63) * 2))])), (&(((half_t*)buf_dyn_shmem)[((i_5 * 256) + (((int)threadIdx.x) * 2))])));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 32768, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(16, 8, 4), dim3(128, 1, 1), 32768, stream>>>(A, B, C);

    return 0;
}

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#include <cuda_fp8.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/gemm_sm90.h>
#include <cuda.h>
#include <cuda_runtime.h>
extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc, float* __restrict__ scales_a, float* __restrict__ scales_b);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc, float* __restrict__ scales_a, float* __restrict__ scales_b) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  float C_local_accum[128];
  __shared__ uint64_t _mbarrier[18];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 128);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
    tl::mbarrier_init(_mbarrier[10], 128);
    tl::mbarrier_init(_mbarrier[11], 128);
    tl::mbarrier_init(_mbarrier[12], 128);
    tl::mbarrier_init(_mbarrier[13], 128);
    tl::mbarrier_init(_mbarrier[14], 128);
    tl::mbarrier_init(_mbarrier[15], 128);
    tl::mbarrier_init(_mbarrier[16], 128);
    tl::mbarrier_init(_mbarrier[17], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    for (int k = 0; k < 64; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 3) + 8)], (((k & 7) >> 2) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 3)], 16384);
        tl::tma_load(A_desc, _mbarrier[(k & 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((k & 3) * 16384) + 67584)])), (k * 128), (((int)blockIdx.y) * 128));
        tl::mbarrier_expect_tx(_mbarrier[(k & 3)], 16384);
        tl::tma_load(A_desc, _mbarrier[(k & 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((k & 3) * 16384) + 67584)])), (k * 128), (((int)blockIdx.y) * 128));
        tl::mbarrier_expect_tx(_mbarrier[(k & 3)], 16384);
        tl::tma_load(B_desc, _mbarrier[(k & 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((k & 3) * 16384) + 2048)])), (k * 128), (((int)blockIdx.x) * 128));
      }
      tl::mbarrier_arrive(_mbarrier[(k & 3)]);
      tl::mbarrier_wait(_mbarrier[((k & 3) + 12)], (((k & 7) >> 2) ^ 1));
      ((float*)buf_dyn_shmem)[((((k & 3) * 128) + ((int)threadIdx.x)) - 128)] = (scales_a[((((((int)blockIdx.y) * 8192) + (((int)threadIdx.x) * 64)) + k) - 8192)] * scales_b[((((int)blockIdx.x) * 64) + k)]);
      tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(_mbarrier[((k & 3) + 4)]);
      tl::mbarrier_arrive(_mbarrier[((k & 3) + 4)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      *(float2*)(C_local_accum + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 & 3)], ((k_1 & 7) >> 2));
      tl::gemm_ss<128, 128, 128, 4, 1, 0, 1, 0, true>((&(((fp8_e4_t*)buf_dyn_shmem)[(((k_1 & 3) * 16384) + 67584)])), (&(((fp8_e4_t*)buf_dyn_shmem)[(((k_1 & 3) * 16384) + 2048)])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 3) + 8)]);
      tl::mbarrier_wait(_mbarrier[((k_1 & 3) + 4)], ((k_1 & 7) >> 2));
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        float2 __1;
          float2 v_ = *(float2*)(C_local_accum + (i_2 * 2));
          float2 __2;
            float2 v__1 = *(float2*)(C_local + (i_2 * 2));
            float2 v__2 = make_float2(((float*)buf_dyn_shmem)[((((((k_1 & 3) * 128) + ((i_2 >> 5) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)buf_dyn_shmem)[((((((k_1 & 3) * 128) + ((i_2 >> 5) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            __2.x = (v__1.x*v__2.x);
            __2.y = (v__1.y*v__2.y);
          __1.x = (v_.x+__2.x);
          __1.y = (v_.y+__2.y);
        *(float2*)(C_local_accum + (i_2 * 2)) = __1;
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[((k_1 & 3) + 12)]);
      #pragma unroll
      for (int i_3 = 0; i_3 < 64; ++i_3) {
        *(float2*)(C_local + (i_3 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
    }
    tl::syncthreads_partial(_mbarrier[16]);
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      *(float2*)(((float*)buf_dyn_shmem) + ((((((((i_4 >> 5) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_4 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((i_4 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16896)) = *(float2*)(C_local_accum + (i_4 * 2));
    }
    tl::fence_proxy_async();
    tl::syncthreads_partial(_mbarrier[17]);
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(C_desc, (&(((float*)buf_dyn_shmem)[16896])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 133120);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 133120, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(fp8_e4_t* __restrict__ A, fp8_e4_t* __restrict__ B, float* __restrict__ C, float* __restrict__ scales_a, float* __restrict__ scales_b, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)0;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {8192,1024};
	cuuint64_t A_desc_globalStride[2]= {1,8192};
	cuuint32_t A_desc_boxDim[2]= {128,128};
	cuuint32_t A_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

	if (A_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor A_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap B_desc;
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)0;
	cuuint32_t B_desc_tensorRank= 2;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[2]= {8192,1024};
	cuuint64_t B_desc_globalStride[2]= {1,8192};
	cuuint32_t B_desc_boxDim[2]= {128,128};
	cuuint32_t B_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

	if (B_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor B_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap C_desc;
	CUtensorMapDataType C_desc_type= (CUtensorMapDataType)7;
	cuuint32_t C_desc_tensorRank= 2;
	void *C_desc_globalAddress= C;
	cuuint64_t C_desc_globalDim[2]= {1024,1024};
	cuuint64_t C_desc_globalStride[2]= {4,4096};
	cuuint32_t C_desc_boxDim[2]= {128,128};
	cuuint32_t C_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave C_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle C_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion C_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill C_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult C_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, C_desc_globalStride + 1, C_desc_boxDim, C_desc_elementStrides, C_desc_interleave, C_desc_swizzle, C_desc_l2Promotion, C_desc_oobFill);

	if (C_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor C_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 133120, stream>>>(A_desc, B_desc, C_desc, scales_a, scales_b);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}

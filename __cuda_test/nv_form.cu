#include <cuda_runtime.h>
#include <iostream>

__global__ void metrics_kernel(float *src, float *dst, int count, int N) {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (id_x >= count) {
    return;
  }

  dst[id_x * N] = src[id_x * N];
}

int main(int argc, char *argv[]) {
  int N = 1;

  if (argc == 2) {
    N = atoi(argv[1]);
  }

  const int kCount = 32;

  // alloc N times the memory space for stride load
  int size = kCount * sizeof(float) * N;

  float *src = static_cast<float *>(malloc(size));
  float *dst = static_cast<float *>(malloc(size));

  for (int i = 0; i < kCount * N; ++i) {
    src[i] = i;
  }

  float *src_dev = nullptr;
  float *dst_dev = nullptr;

  cudaMalloc(&src_dev, size);
  cudaMalloc(&dst_dev, size);

  cudaMemcpy(src_dev, src, size, cudaMemcpyHostToDevice);

  dim3 block(32, 1);
  dim3 grid(1, 1);

  metrics_kernel<<<grid, block>>>(src_dev, dst_dev, kCount, N);

  cudaStreamSynchronize(0);

  cudaFree(src_dev);
  cudaFree(dst_dev);

  free(src);
  free(dst);

  return 0;
}
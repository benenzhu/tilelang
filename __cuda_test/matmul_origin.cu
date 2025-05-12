#include <iostream>
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <cmath>   // For fabs()
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// CUDA运行时API
#include <cuda_runtime.h>

// 用于检查CUDA API调用错误的宏
#define CUDA_CHECK(err)                                                               \
    do {                                                                              \
        cudaError_t err_ = (err);                                                     \
        if (err_ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "      \
                      << cudaGetErrorString(err_) << std::endl;                       \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

static constexpr int  N = 64 * 50;
static constexpr int TILE_DIM = 32;
// CUDA Kernel
// 计算 C = A * B，其中 A, B, C 都是 N x N 的方阵
// 每个线程计算输出矩阵 C 的一个元素
// 
// 

// A[N, TILE_DIM] B[TILE_DIM, N]
// __global__ void sharedABMultiply(float *a, float* b, float *c)
// {
//     __shared__ float aTile[TILE_DIM][TILE_DIM],
//                      bTile[TILE_DIM][TILE_DIM];
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0.0f;
//     aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
//     bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
//     __syncthreads();
//     for (int i = 0; i < TILE_DIM; i++) {
//         sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
//         sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
//     }
//     c[row*N+col] = sum;
// }

__global__ void sharedABMultiply(float *a, float* b, float *c)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float bTile[TILE_DIM][TILE_DIM];
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
    __syncthreads();
    for(int i = 0; i < TILE_DIM; i++){
        sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
    }
    c[row * N + col] = sum;
}
__global__ void simpleMultiply(float *a, float* b, float *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row * TILE_DIM + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}


__global__ void coalescedMultiply(float *a, float* b, float *c)
{
    __shared__ float aTile[TILE_DIM * TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y * TILE_DIM + threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    // if(threadIdx.y == 0 && blockIdx.y == 0 && blockIdx. == 0) {
    //     printf("%d %d\n", threadIdx.x, threadIdx.y * TILE_DIM + threadIdx.x);
    // }
    __syncwarp();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y * TILE_DIM + i]* b[i*N+col];
    }
    c[row*N+col] = sum;
}


// 用于在CPU上执行矩阵乘法以验证结果的辅助函数
void matrixMultiplyCPU(const thrust::host_vector<float>& a, const thrust::host_vector<float>& b, thrust::host_vector<float>& c) {
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < TILE_DIM; ++i) {
                sum += a[row * TILE_DIM + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}



void matrixTransMultiplyCPU(const thrust::host_vector<float>& a, thrust::host_vector<float>& c) {
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < TILE_DIM; ++i) {
                sum += a[row * TILE_DIM + i] * a[col * TILE_DIM + i];
            }
            c[row * N + col] = sum;
        }
    }
}

__global__ void simpleTransMultiply(float *a, float *c, int M)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
    }
    c[row*M+col] = sum;
}

__global__ void coalescedTransMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}

__global__ void coalescedPadTransMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM + 1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}
__global__ void coalescedPadSlowTransMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM + 1],
                     transposedTile[TILE_DIM][TILE_DIM + 1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}

// __global__ void sharedTransMultiply(float *a, float *c, int M)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ float aTile[TILE_DIM][TILE_DIM];
//     __shared__ float bTile[TILE_DIM]
    
//     float sum = 0.0f;
//     for (int i = 0; i < TILE_DIM; i++) {
//         sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
//     }
//     c[row*M+col] = sum;
// }


int main() {
    const int matrix_size_bytes = N * N * sizeof(float);

    thrust::host_vector<float> h_a(N * TILE_DIM);
    thrust::host_vector<float> h_b(TILE_DIM * N);
    thrust::host_vector<float> h_c_gpu(N * N); // 用于存储GPU计算结果
    thrust::host_vector<float> h_c_cpu(N * N); // 用于存储CPU计算结果 (验证用)

    // 用随机数初始化矩阵 A 和 B
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < TILE_DIM * N; ++i) {
        // h_a[i] = 1;
        // h_b[i] = 1;
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // --- 2. 将数据转移到设备 ---
    thrust::device_vector<float> dv_a = h_a;  // 自动拷贝到GPU
    thrust::device_vector<float> dv_b = h_b;
    thrust::device_vector<float> dv_c(N * N);

    // --- 3. 获取原始指针用于kernel调用 ---
    float* d_a = thrust::raw_pointer_cast(dv_a.data());
    float* d_b = thrust::raw_pointer_cast(dv_b.data());
    float* d_c = thrust::raw_pointer_cast(dv_c.data());
    // CUDA_CHECK(cudaMalloc((void**)&d_a, ab_size_byts));
    // CUDA_CHECK(cudaMalloc((void**)&d_b, ab_size_byts));
    // CUDA_CHECK(cudaMalloc((void**)&d_c, matrix_size_bytes));

    // --- 3. 将数据从主机复制到设备 ---
    // CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), ab_size_byts, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), ab_size_byts, cudaMemcpyHostToDevice));

    // --- 4. Kernel启动配置 ---
    // 定义每个block的线程数。通常选择16x16=256或32x32=1024个线程
    // 这里使用16x16，因为这是很多GPU架构上性能较好的选择
    dim3 blockSize(TILE_DIM, TILE_DIM);

    // 计算grid的维度，确保覆盖整个N x N矩阵
    // (N + D - 1) / D 是一种向上取整的常用方法
    dim3 gridSize(N /  TILE_DIM, N / TILE_DIM);

    std::cout << "Matrix dimensions: " << N << "x" << N << std::endl;
    std::cout << "Block dimensions: " << blockSize.x << "x" << blockSize.y << std::endl;
    std::cout << "Grid dimensions: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Total threads: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;

    // --- 5. 执行Kernel ---
    std::cout << "Launching CUDA kernel..." << std::endl;
    simpleMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    coalescedMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    sharedABMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    // 检查Kernel启动是否有错误
    CUDA_CHECK(cudaGetLastError());
    // 同步设备，确保Kernel执行完毕
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "CUDA kernel finished." << std::endl; 

    // return 0;
    // --- 6. 将结果从设备复制回主机 ---
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, matrix_size_bytes, cudaMemcpyDeviceToHost));

    // --- 7. 验证结果 (可选但推荐) ---
    std::cout << "Verifying results on CPU..." << std::endl;
    matrixMultiplyCPU(h_a, h_b, h_c_cpu);

    bool match = true;
    float tolerance= 1e-4; // 浮点数比较容差
    int cnt = 0;
    std::cout << "compare here...";
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(h_c_gpu[i] - h_c_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i / N << "," << i % N << ": GPU=" << h_c_gpu[i]
                      << ", CPU=" << h_c_cpu[i] << std::endl;
            match = false;
            cnt++;
            // break;
        }
        if(cnt > 10) break;
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results DO NOT match!" << std::endl;
    }
    
    /**--------------------------------------------------------------trans */
    simpleTransMultiply<<<gridSize, blockSize>>>(d_a, d_c, N);
    coalescedTransMultiply<<<gridSize, blockSize>>>(d_a, d_c, N);
    coalescedPadTransMultiply<<<gridSize, blockSize>>>(d_a, d_c, N);
    coalescedPadSlowTransMultiply<<<gridSize, blockSize>>>(d_a, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "CUDA kernel finished." << std::endl; 

    // return 0;
    // --- 6. 将结果从设备复制回主机 ---
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, matrix_size_bytes, cudaMemcpyDeviceToHost));

    // --- 7. 验证结果 (可选但推荐) ---
    std::cout << "Verifying results on CPU..." << std::endl;
    matrixTransMultiplyCPU(h_a, h_c_cpu);

    match = true;
    tolerance= 1e-4; // 浮点数比较容差
    cnt = 0;
    std::cout << "compare here...";
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(h_c_gpu[i] - h_c_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i / N << "," << i % N << ": GPU=" << h_c_gpu[i]
                      << ", CPU=" << h_c_cpu[i] << std::endl;
            match = false;
            cnt++;
            // break;
        }
        if(cnt > 10) break;
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results DO NOT match!" << std::endl;
    }




    std::cout << "Program finished." << std::endl;
    return 0;
}

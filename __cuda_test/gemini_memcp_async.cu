#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <numeric> // For std::accumulate
#include <cmath>   // For std::round

// CUDA runtime
#include <cuda_runtime.h>
#include <cstdint> // For uint64_t

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

// Custom data types
struct MyInt4 { // 16 bytes
  int v0, v1, v2, v3;
};

// --- Kernels (Provided in the question) ---
template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
  }

  uint64_t clock_end = clock64();

  // Only one thread needs to write this if we want total block time.
  // But the current atomicAdd sums up all thread's individual processing times.
  // To get an effective block time, we'll divide this sum by blockDim.x later.
  if (threadIdx.x == 0 && blockIdx.x == 0) { // Ensure only one block contributes if grid > 1
      atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                clock_end - clock_start);
  }
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);
  uint64_t clock_start = clock64();
  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], 
                            sizeof(T)); // Copy one element of type T per thread per iteration
  }
  __pipeline_commit();
  __pipeline_wait_prior(0); // Wait for all prior stages (0 means all previous stages)
  uint64_t clock_end = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) { // Ensure only one block contributes if grid > 1
      atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                clock_end - clock_start);
  }
}


// Helper function to run benchmark for a given type and configuration
template <typename T>
void run_test(const std::string& type_name, size_t element_size_bytes,
              size_t total_copy_bytes_per_block,
              int block_size, int repetitions, double gpu_clock_rate_ghz,
              std::vector<double>& sync_times_ms, std::vector<double>& async_times_ms,
              std::vector<double>& sync_bws, std::vector<double>& async_bws) {

    if (total_copy_bytes_per_block < block_size * element_size_bytes && total_copy_bytes_per_block % (block_size * element_size_bytes) != 0) {
        std::cout << std::setw(10) << type_name
                  << std::setw(12) << (total_copy_bytes_per_block / 1024.0)
                  << "  SKIPPED (total_copy_bytes < block_size * element_size or not multiple)" << std::endl;
        sync_times_ms.push_back(-1.0); async_times_ms.push_back(-1.0);
        sync_bws.push_back(-1.0); async_bws.push_back(-1.0);
        return;
    }
    
    size_t copy_count_per_thread = total_copy_bytes_per_block / (block_size * element_size_bytes);
    if (copy_count_per_thread == 0) {
         std::cout << std::setw(10) << type_name
                  << std::setw(12) << std::fixed << std::setprecision(2) << (total_copy_bytes_per_block / 1024.0)
                  << "  SKIPPED (copy_count_per_thread is 0)" << std::endl;
        sync_times_ms.push_back(-1.0); async_times_ms.push_back(-1.0);
        sync_bws.push_back(-1.0); async_bws.push_back(-1.0);
        return;
    }


    size_t num_elements_total = block_size * copy_count_per_thread; // Elements for one block
    size_t global_mem_size_bytes = num_elements_total * element_size_bytes;

    T* h_data = new T[num_elements_total];
    for(size_t i = 0; i < num_elements_total; ++i) {
        // Simple initialization
        char* p = reinterpret_cast<char*>(&h_data[i]);
        for(size_t b = 0; b < element_size_bytes; ++b) p[b] = (i+b) % 256;
    }

    T* d_global;
    uint64_t* d_clock;

    CUDA_CHECK(cudaMalloc(&d_global, global_mem_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_clock, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_global, h_data, global_mem_size_bytes, cudaMemcpyHostToDevice));

    dim3 grid_dim(1); // We are measuring per-block performance
    dim3 block_dim(block_size);
    size_t shared_mem_bytes = total_copy_bytes_per_block; // This is crucial

    // --- Sync Kernel ---
    double total_sync_cycles = 0;
    uint64_t h_clock_val;

    // Warm-up
    CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));
    pipeline_kernel_sync<T><<<grid_dim, block_dim, shared_mem_bytes>>>(d_global, d_clock, copy_count_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));
        pipeline_kernel_sync<T><<<grid_dim, block_dim, shared_mem_bytes>>>(d_global, d_clock, copy_count_per_thread);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_clock_val, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total_sync_cycles += h_clock_val;
    }
    double avg_sync_cycles = static_cast<double>(total_sync_cycles) / repetitions;
    // The kernel's atomicAdd sums cycles from all threads. To get effective block time in cycles,
    // we should ideally take the max over threads.
    // Since only thread 0 writes, this is clock_end - clock_start for thread 0.
    // This is the time for thread 0 to complete its loops.
    double sync_time_ms = (avg_sync_cycles / (gpu_clock_rate_ghz * 1e9)) * 1000.0;
    double sync_bw_gbs = (total_copy_bytes_per_block / (sync_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    sync_times_ms.push_back(sync_time_ms);
    sync_bws.push_back(sync_bw_gbs);

    // --- Async Kernel ---
    uint64_t total_async_cycles = 0;
    // Warm-up
    CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));
    pipeline_kernel_async<T><<<grid_dim, block_dim, shared_mem_bytes>>>(d_global, d_clock, copy_count_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));
        pipeline_kernel_async<T><<<grid_dim, block_dim, shared_mem_bytes>>>(d_global, d_clock, copy_count_per_thread);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_clock_val, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // printf("cycle: %llf\n", h_clock_val);
        total_async_cycles += h_clock_val;
    }
    double avg_async_cycles = static_cast<double>(total_async_cycles) / repetitions;
    double async_time_ms = (avg_async_cycles / (gpu_clock_rate_ghz * 1e9)) * 1000.0;
    double async_bw_gbs = (total_copy_bytes_per_block / (async_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    async_times_ms.push_back(async_time_ms);
    async_bws.push_back(async_bw_gbs);

    // Cleanup
    CUDA_CHECK(cudaFree(d_global));
    CUDA_CHECK(cudaFree(d_clock));
    delete[] h_data;

    // Print individual row
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(10) << type_name
              << std::setw(12) << (total_copy_bytes_per_block / 1024.0) // KB
              << std::setw(15) << sync_time_ms
              << std::setw(15) << async_time_ms
              << std::setw(15) << sync_bw_gbs
              << std::setw(15) << async_bw_gbs
              << std::endl;
}


int main() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    double gpu_clock_rate_ghz = static_cast<double>(props.clockRate) / (1000.0 * 1000.0); // kHz to GHz

    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "GPU Clock Rate: " << std::fixed << std::setprecision(2) << gpu_clock_rate_ghz << " GHz" << std::endl;
    std::cout << "Max Shared Memory Per Block: " << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    // Note: Max dynamic shared memory is props.sharedMemPerBlock - static shared memory used by kernel.
    // We are using dynamic shared memory.
    // On some architectures, you can opt-in for more shared memory per SM (e.g. 96KB vs 32KB cache on Ampere)
    // This might require cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
    // For simplicity, we'll stick to standard limits.

    int block_size = 256; // A common block size
    int repetitions = /*for fast 100*/1; // Number of times to run each kernel for averaging
    std::cout << "Block Size: " << block_size << " threads" << std::endl;
    std::cout << "Repetitions per test: " << repetitions << std::endl << std::endl;

    std::cout << std::setw(10) << "Elem Type"
              << std::setw(12) << "Copy/Blk(KB)"
              << std::setw(15) << "Sync Time(ms)"
              << std::setw(15) << "Async Time(ms)"
              << std::setw(15) << "Sync BW(GB/s)"
              << std::setw(15) << "Async BW(GB/s)"
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    std::vector<size_t> copy_sizes_bytes;
    // From 512 B up to 48 KB
    // std::vector<int> size = { 1, 2, 4, 8, 16, 32, 48};
    for (size_t kb_size : {/* for fast ncu1, 2, 4, 8, 16, 32,*/ 48}) { // KB values
         if (kb_size * 1024 <= props.sharedMemPerBlock) { // Ensure we don't exceed device limits
            copy_sizes_bytes.push_back(static_cast<size_t>(kb_size * 1024));
         } else if (!copy_sizes_bytes.empty() && copy_sizes_bytes.back() < props.sharedMemPerBlock && kb_size * 1024 > props.sharedMemPerBlock) {
            // Add the max possible if we jumped over it
            copy_sizes_bytes.push_back(props.sharedMemPerBlock);
         }
    }
    // Ensure the absolute max shared memory is tested if not already included and it's a power of 2 or common value.
    // For simplicity, we will use the list above. If props.sharedMemPerBlock is e.g. 49152 (48KB), it will be included.

    // Store all results for potential later processing if needed
    std::vector<double> all_sync_times, all_async_times, all_sync_bws, all_async_bws;

    // Test with int (4 Bytes)
    for (size_t cs : copy_sizes_bytes) {
        run_test<int>("int (4B)", sizeof(int), cs, block_size, repetitions, gpu_clock_rate_ghz,
                      all_sync_times, all_async_times, all_sync_bws, all_async_bws);
    }
    std::cout << std::string(82, '-') << std::endl;

    // Test with long long (8 Bytes)
    for (size_t cs : copy_sizes_bytes) {
        run_test<long long>("llong (8B)", sizeof(long long), cs, block_size, repetitions, gpu_clock_rate_ghz,
                            all_sync_times, all_async_times, all_sync_bws, all_async_bws);
    }
    std::cout << std::string(82, '-') << std::endl;

    // Test with MyInt4 (16 Bytes)
    for (size_t cs : copy_sizes_bytes) {
        run_test<MyInt4>("MyInt4(16B)", sizeof(MyInt4), cs, block_size, repetitions, gpu_clock_rate_ghz,
                         all_sync_times, all_async_times, all_sync_bws, all_async_bws);
    }
    std::cout << std::string(82, '-') << std::endl;

    return 0;
}

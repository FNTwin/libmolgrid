/*
 * common.h
 *
 * Utility functions and definitions
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#ifdef CPU_ONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER
#endif

//called in device code to perform a parallel operation
#ifndef __CUDACC__
#define blockIdx.x
#define LMG_CUDA_KERNEL_LOOP(i, n) \
    #define __global__
    int threadIdxX;
    int blockIdxX;
    int blockDimX;
    int gridDimX;
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdxX * blockDimX + threadIdxX; \
       i < (n); \
       i += blockDimX * gridDimX)
#else
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
#endif
// CUDA: use 512 threads per block
#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM 8
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

// CUDA: number of blocks for threads.
#define LMG_GET_BLOCKS(N) ((unsigned(N) + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)
// CUDA: combined with GET_BLOCKS, number of threads
#define LMG_GET_THREADS(N) min(N,LMG_CUDA_NUM_THREADS)

#ifndef __CUDA_ARCH__
#define LMG_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) {                                          \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error); throw std::runtime_error(std::string("CUDA Error: ")+cudaGetErrorString(error)); } \
  } while (0)
#else
// probably don't want to make API calls on the device.
#define LMG_CUDA_CHECK(condition) condition
#endif


#endif /* COMMON_H_ */

#pragma once


#ifdef STRUCTLIGHTCUDA_EXPORTS
#define STRUCTLIGHTCUDA_API __declspec(dllexport)
#else
#define STRUCTLIGHTCUDA_API __declspec(dllimport)
#endif
const int CUDA_NUM_THREADS = 512;
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)
#define STREAM_NUM 2
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
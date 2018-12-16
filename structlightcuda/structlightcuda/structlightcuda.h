// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 STRUCTLIGHTCUDA_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// STRUCTLIGHTCUDA_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifndef _SLIGHTCUDA__H 
#define _SLIGHTCUDA__H

#ifdef STRUCTLIGHTCUDA_EXPORTS
#define STRUCTLIGHTCUDA_API __declspec(dllexport)
#else
#define STRUCTLIGHTCUDA_API __declspec(dllimport)
#endif
const int CUDA_NUM_THREADS = 1024;
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)
#define STREAM_NUM 2
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

extern "C" STRUCTLIGHTCUDA_API 
int reconstruct3D_gpu(double* lPoints, double* rPoints, float *pCloud,int pointsNum
,float *pA1, float *pA2, float *pAd1, float *pAd2, float *pAD);

extern "C" STRUCTLIGHTCUDA_API int opengpu();
extern "C" STRUCTLIGHTCUDA_API int initmemorygpu(int size);
extern "C" STRUCTLIGHTCUDA_API int freememorygpu();
extern "C" STRUCTLIGHTCUDA_API int closegpu();
#endif // !_SLIGHTCUDA__H 





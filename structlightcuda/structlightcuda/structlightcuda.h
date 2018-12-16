// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� STRUCTLIGHTCUDA_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// STRUCTLIGHTCUDA_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
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





// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 STRUCTLIGHTCUDA_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// STRUCTLIGHTCUDA_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifndef _SLIGHTCUDA__H 
#define _SLIGHTCUDA__H
#include "cuda_gobal.h"
#define HostToDevice 1
#define DeviceToHost 2
/*****************only restruct****************************/
extern "C" STRUCTLIGHTCUDA_API
int reconstruct3D_gpu(double* lPoints, double* rPoints, float *pCloud, int pointsNum
	, float *pA1, float *pA2, float *pAd1, float *pAd2, float *pAD);
extern "C" STRUCTLIGHTCUDA_API int initrecmemorygpu(int size);
extern "C" STRUCTLIGHTCUDA_API int freerecmemorygpu();
/****************all the structlight***********************/
extern "C" STRUCTLIGHTCUDA_API double* doublememorygpu(int size,double** p);
extern "C" STRUCTLIGHTCUDA_API float* floatmemorygpu(int size,float **p);
extern "C" STRUCTLIGHTCUDA_API unsigned char* ucharmemorygpu(int size, unsigned char **p);
extern "C" STRUCTLIGHTCUDA_API void freememorygpu_d(double *p);
extern "C" STRUCTLIGHTCUDA_API void freememorygpu_f(float *p);
extern "C" STRUCTLIGHTCUDA_API void freememorygpu_c(unsigned char *p);
extern "C" STRUCTLIGHTCUDA_API double * testadd(double *pg, double *res, int num);
extern "C" STRUCTLIGHTCUDA_API double* doublememcpygpu(int size, double *dirp, double *dp, int model);
extern "C" STRUCTLIGHTCUDA_API float* floatmemcpygpu(int size, float *dirp, float *dp, int model);
extern "C" STRUCTLIGHTCUDA_API unsigned char* ucharmemcpygpu(int size, unsigned char *dirp, unsigned char *dp, int model);
extern "C" STRUCTLIGHTCUDA_API void decodeunwarp_gpu(unsigned char *point, int imgnum, int pointnum, double *outputx, double *outputy);
extern "C" STRUCTLIGHTCUDA_API void decodewarp_gpu(double *unwarp1, double *unwarp2, double *unwarp3, double *warpped, double wave1, double wave2, double wave3, int pointnum);
extern "C" STRUCTLIGHTCUDA_API void CorrectPoints_gpu(double *unwarpx, double *unwrapY, int num, int xpixels, int ypixels, double *rpoints);
extern "C" STRUCTLIGHTCUDA_API int reconstruct_gpu(double* lPoints, double* rPoints, float *pCloud, int pointsNum
	, float *cpA1, float *cpA2, float *cpAd1, float *cpAd2, float *cpAD);
/************************************************/
extern "C" STRUCTLIGHTCUDA_API int opengpu();
extern "C" STRUCTLIGHTCUDA_API int closegpu();
#endif // !_SLIGHTCUDA__H 





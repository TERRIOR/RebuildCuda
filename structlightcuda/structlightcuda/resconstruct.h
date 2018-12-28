#pragma once
#include "cuda_gobal.h"
extern "C" STRUCTLIGHTCUDA_API
int reconstruct3D_gpu(double* lPoints, double* rPoints, float *pCloud, int pointsNum
	, float *pA1, float *pA2, float *pAd1, float *pAd2, float *pAD);
extern "C" STRUCTLIGHTCUDA_API int initrecmemorygpu(int size);
extern "C" STRUCTLIGHTCUDA_API int freerecmemorygpu();
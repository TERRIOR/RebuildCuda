#include "cuda_runtime.h" 
#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"    
#include "structlightcuda.h"
#include "iostream"
#include "stdio.h"

__device__ double *dev_pointl = 0;
__device__ double *dev_pointr = 0;
__device__ float *dev_p = 0;
__constant__ float pA1[9], pA2[9], pAd1[3], pAd2[3], pAD[3];

__global__ void pointKernel(double *lPoints,double *rPoints,float *pCloud ,int pointnum)
{
	//
	float U1, U2, V1, V2;
	CUDA_KERNEL_LOOP(i, pointnum) {

		U1 = lPoints[2 * i];
		V1 = lPoints[2 * i + 1];
		U2 = rPoints[2 * i ];
		V2 = rPoints[2 * i + 1];


		if (U2 > 1280 || U2 < 0 || V2 > 800 || V2 < 0)
		{
			pCloud[3*i] = 0;
			pCloud[3*i+1] = 0;
			pCloud[3*i+2] = 0;
			//respoint[0]=0;
			//respoint[1] = 0;
			//respoint[2] = 0;
		}
		else {
			
			float res1[3];
			float res2[3];
			float B1[3];
			float B2[3];
			float B[6];
			float BN[6];
			float BxBN[4];
			float invBxBN[4];
			float invB[6];
			float S[2];
			float det;
			float vdet[4];
			float Btemp1;
			float Btemp2;
			
			//ʹ��������Ŵ���S�ķ�����ֻ��Ҫ�Զ�ά��������
			//��B��B��ת��
			for (int var = 0; var <3; ++var) {
	
				Btemp1 = pA1[var * 3] * U1 + pA1[var * 3 + 1] * V1 + pA1[var * 3 + 2];
				Btemp2 = pA2[var * 3] * U2 + pA2[var * 3 + 1] * V2 + pA2[var * 3 + 2];
				B1[var] = Btemp1;
				B2[var] = Btemp2;
				B[var * 2] = Btemp1;
				B[var * 2 + 1] = -Btemp2;
				BN[var] = Btemp1;
				BN[var + 3] = -Btemp2;
			}

			//B��ת����B���
			BxBN[0] = BN[0] * B[0] + BN[1] * B[2] + BN[2] * B[4];
			BxBN[1] = BN[0] * B[1] + BN[1] * B[3] + BN[2] * B[5];
			BxBN[2] = BN[3] * B[0] + BN[4] * B[2] + BN[5] * B[4];
			BxBN[3] = BN[3] * B[1] + BN[4] * B[3] + BN[5] * B[5];
		
			//��������ʽ
			det = 1 / (BxBN[0] * BxBN[3] - BxBN[1] * BxBN[2]);
			//��ά��������
			invBxBN[0] = BxBN[3] * det;
			invBxBN[1] = -BxBN[1] * det;
			invBxBN[2] = -BxBN[2] * det;
			invBxBN[3] = BxBN[0] * det;
			
			//BxBN�������B���
			for (int var = 0; var < 3; ++var) {
				invB[var] = invBxBN[0] * BN[var] + invBxBN[1] * BN[var + 3];
				invB[var + 3] = invBxBN[2] * BN[var] + invBxBN[3] * BN[var + 3];
			}
			//��������Ŵ�ϵ��S
			S[0] = invB[0] * pAD[0] + invB[1] * pAD[1] + invB[2] * pAD[2];
			S[1] = invB[3] * pAD[0] + invB[4] * pAD[1] + invB[5] * pAD[2];
			//�ֱ�������꣬ȡƽ��
			//float xyz[3];
			//pxyz xyzp;

			for (int var = 0; var < 3; ++var) {
				res1[var] = B1[var] * S[0] - pAd1[var];
				res2[var] = B2[var] * S[1] - pAd2[var];
				//respoint[ var] = (res1[var] + res2[var]) / 2;
				pCloud[3 * i + var] = (res1[var] + res2[var]) / 2;
			}
		}
		//__syncthreads();
		//pCloud[3 * i] = respoint[0];
		//pCloud[3 * i+1] = respoint[1];
		//pCloud[3 * i+2] = respoint[2];
		//printf("3:%f ", pCloud[3 * i]);
		//std::cout << << std::endl;
	}
}

STRUCTLIGHTCUDA_API int reconstruct3D_gpu(double* lPoints, double* rPoints, float *pCloud,int pointsNum
	, float *cpA1, float *cpA2, float *cpAd1, float *cpAd2, float *cpAD) {
	int result = -1;
	static int count = 0;
	cudaError_t cudaStatus;
	// �������ڴ渴�����ݵ�GPU�ڴ���.  
	cudaStatus = cudaMemcpy(dev_pointr, rPoints,2 * pointsNum * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 6;
		goto Error;
	}
	if (count==0)
	{
		cudaStatus = cudaMemcpy(dev_pointl, lPoints,2 * pointsNum * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			result = 6;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pA1, cpA1, 9* sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 7;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pA2, cpA2, 9 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 8;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd1, cpAd1,3* sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 9;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd2, cpAd2,3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 10;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAD, cpAD, 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 11;
			goto Error;
		}
		count = 1;
	}
	

	pointKernel <<< GET_BLOCKS(pointsNum), CUDA_NUM_THREADS >>>(dev_pointl,dev_pointr,dev_p,pointsNum);

	// ����cudaDeviceSynchronize�ȴ�GPU�ں˺���ִ����ɲ��ҷ����������κδ�����Ϣ  
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	result = 7;
	//	goto Error;
	//}

	// ��GPU�ڴ��и������ݵ������ڴ���  
	cudaStatus = cudaMemcpy(pCloud, dev_p, 3*pointsNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		result = 12;
		goto Error;
	}

	result = 0;


Error:
	//�ͷ��豸�б�����ռ�ڴ�  
	return result;
}


STRUCTLIGHTCUDA_API int initmemorygpu(int size) {
	cudaError_t cudaStatus;
	int result=0;
	static int inited = 0;
	if (inited==1)
	{
		return 1;
	}
	// ѡ���������е�GPU  

	if (opengpu()<0) {
		printf("cannot open cuda");
		result = 1;
	}
	// ��GPU��Ϊ����dev_a��dev_b��dev_c�����ڴ�ռ�.  
	cudaStatus = cudaMalloc((void**)&dev_p, 3 * size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		result = 2;
	}
	cudaStatus = cudaMalloc((void**)&dev_pointr, 2 * size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		result = 4;
	}
	cudaStatus = cudaMalloc((void**)&dev_pointl, 2 * size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		result = 4;
	}
	inited = 1;
	return result;
}
STRUCTLIGHTCUDA_API int freememorygpu() {
	static int freed=0;
	if (freed == 1) {
		return 1;
	}
	if (dev_p!=0|| dev_pointr != 0|| dev_pointl != 0)
	{
		cudaFree(dev_p);
		cudaFree(dev_pointl);
		cudaFree(dev_pointr);
	}
	freed = 1;
	return 1;
}

STRUCTLIGHTCUDA_API int opengpu() {
	static int opened = 0;
	if (opened == 0) {
		cudaError_t cudaStatus;
		// ѡ���������е�GPU  
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			printf("cannot open cuda");
			return -1;
		}
		printf("open cuda successfully");
		opened = 1;
	}
	return 1;
}

STRUCTLIGHTCUDA_API int closegpu() {
	static int closed = 0;
	if (closed == 0) {
		cudaError_t cudaStatus;
		// ѡ���������е�GPU  
		cudaStatus = cudaFree(0);
		if (cudaStatus != cudaSuccess) {
			printf("cannot close cuda");
			return -1;
		}
		closed = 1;
		printf("close the cuda successfully");
	}
	return 1;
}
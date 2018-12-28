#include "cuda_runtime.h" 
#include "cuda.h"
#include "math.h"
//#include "math_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"    
#include "structlightcuda.h"

#include "stdio.h"
#include "cuda_gobal.h"

__device__ double *dev_pointl = 0;
__device__ double *dev_pointr = 0;
__device__ float *dev_p = 0;
__constant__ float pA1[9], pA2[9], pAd1[3], pAd2[3], pAD[3];
__constant__  double pi = 3.1415926535;
__constant__ double sqrt3 = 1.7320508075;
//__constant__ double twopi = 3.1415926535;
__global__ void addKernel(double *p,int pointnum) {
	CUDA_KERNEL_LOOP(i, pointnum) {
		p[i] = 20;
	}
}
__global__ void createCorrectPoints_rkernel(double *unwarpx, double *unwrapY, int num, double pixelsPerTwoPiX, double pixelsPerTwoPiY, double *rpoints)
{
	CUDA_KERNEL_LOOP(r, num)
	{
		rpoints[2 * r] = unwarpx[r] * pixelsPerTwoPiX;
		rpoints[2 * r + 1] = unwrapY[r] * pixelsPerTwoPiY;
	}
}
__global__ void decode_threefrequencykernel(double *unwarp1, double *unwarp2, double *unwarp3, double *warpped, double wave1, double wave2, double wave3, int pointnum)
{

	CUDA_KERNEL_LOOP(var, pointnum) {
		double p12 = wave1*wave2 / (wave2 - wave1);
		double p123 = p12 / (wave3 - p12);
		double value_w12 = (unwarp1[var] - unwarp2[var]) / (2 * pi);
		value_w12 = (value_w12<0) ? value_w12 + 1 : value_w12;
		/***************************************************************************************/
		//解包裹相位wrap_3，1、直接按比例换算123-3，2、计算出k值再换算，3、两者综合作误差补偿
		double temp_unw3 = (value_w12 - unwarp3[var] / (2 * pi));
		temp_unw3 = (temp_unw3<0) ? temp_unw3 + 1 : temp_unw3
		//比例展开wrap3，小跃变
		double value_unw3 = temp_unw3*p123 * 2 * pi;//三频外差结果 包含小数
		double k3 = floor(p123*temp_unw3);//级数
										  //k值展开wrap3,大跃变
		double value_un_k3 = k3 * 2 * pi + unwarp3[var];//三频外差结果 小数替换成原来包裹相位
														  //结合比例展开、k值展开部分进行相位补偿comp_unwrap_3
		double value_un_err3 = value_un_k3 - value_unw3;
		value_un_k3 = (pi<value_un_err3) ? value_un_k3 - 2 * pi : value_un_k3;
		value_un_k3 = (-pi> value_un_err3) ? value_un_k3 + 2 * pi : value_un_k3;
		double correct_unwrap3 = value_un_k3;
		//理论上的unwrap1
		double value_unw1 = correct_unwrap3*wave3 / wave1;
		//校正
		double k1 = round((value_unw1 - unwarp1[var]) / (2 * pi)) - 29;
		warpped[var] = k1*(2 * pi) + unwarp1[var];
		//std::cout<<k1<<std::endl;
	}
}

__global__ void decodeunwarpkernel(unsigned char *point, int imgnum, int pointnum, double *outputx, double *outputy) {

	//double sqrt3 = 0.57735026919;
	CUDA_KERNEL_LOOP(i, pointnum) {
		double imgvalue[6];
		double temp1;
		double temp2;
		double temp3;
		double temp4;
		double temp5;
		double sinx;
		double cosx;
		double siny;
		double cosy;
		
		for (int var2 = 0; var2 < imgnum; ++var2) {
			imgvalue[var2] = (double)point[i*imgnum + var2];
		}

		temp1 = imgvalue[5] - imgvalue[2];
		temp2 = imgvalue[1] - imgvalue[4];
		temp3 = imgvalue[0] - imgvalue[3];
		temp4 = imgvalue[1] + imgvalue[4];
		temp5 = imgvalue[5] + imgvalue[2];
		sinx = sqrt3*(temp1 - temp2) / 6;
		cosx = temp3 / 3 + (temp2 + temp1) / 6;
		siny = sqrt3*(temp5 - temp4) / 6;
		cosy = (imgvalue[0] + imgvalue[3]) / 3 - (temp4 + temp5) / 6;

		sinx = (abs(sinx) > 1e-10) ? sinx : 0;
		cosx = (abs(cosx) > 1e-10) ? cosx : 0;
		siny = (abs(siny) > 1e-10) ? siny : 0;
		cosy = (abs(cosy) > 1e-10) ? cosy : 0;

	    outputx[i]= pi + atan2(sinx , cosx);
		outputy[i] = pi + atan2(siny , cosy);	

	}
}
__global__ void reconKernel(double *lPoints, double *rPoints, float *pCloud, int pointnum)
{
	//
	float U1, U2, V1, V2;
	CUDA_KERNEL_LOOP(i, pointnum) {

		U1 = lPoints[2 * i];
		V1 = lPoints[2 * i + 1];
		U2 = rPoints[2 * i];
		V2 = rPoints[2 * i + 1];
		if (U2 > 1280 || U2 < 0 || V2 > 800 || V2 < 0)
		{
			pCloud[3 * i] = 0;
			pCloud[3 * i + 1] = 0;
			pCloud[3 * i + 2] = 0;
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

			//使用先求出放大倍数S的方法，只需要对二维矩阵求逆
			//求B与B的转置
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

			//B的转置与B相乘
			BxBN[0] = BN[0] * B[0] + BN[1] * B[2] + BN[2] * B[4];
			BxBN[1] = BN[0] * B[1] + BN[1] * B[3] + BN[2] * B[5];
			BxBN[2] = BN[3] * B[0] + BN[4] * B[2] + BN[5] * B[4];
			BxBN[3] = BN[3] * B[1] + BN[4] * B[3] + BN[5] * B[5];

			//计算行列式
			det = 1 / (BxBN[0] * BxBN[3] - BxBN[1] * BxBN[2]);
			//二维矩阵求逆
			invBxBN[0] = BxBN[3] * det;
			invBxBN[1] = -BxBN[1] * det;
			invBxBN[2] = -BxBN[2] * det;
			invBxBN[3] = BxBN[0] * det;

			//BxBN逆矩阵与B相乘
			for (int var = 0; var < 3; ++var) {
				invB[var] = invBxBN[0] * BN[var] + invBxBN[1] * BN[var + 3];
				invB[var + 3] = invBxBN[2] * BN[var] + invBxBN[3] * BN[var + 3];
			}
			//求出两个放大系数S
			S[0] = invB[0] * pAD[0] + invB[1] * pAD[1] + invB[2] * pAD[2];
			S[1] = invB[3] * pAD[0] + invB[4] * pAD[1] + invB[5] * pAD[2];
			//分别求出坐标，取平均
			//float xyz[3];
			//pxyz xyzp;

			for (int var = 0; var < 3; ++var) {
				res1[var] = B1[var] * S[0] - pAd1[var];
				res2[var] = B2[var] * S[1] - pAd2[var];
				//respoint[ var] = (res1[var] + res2[var]) / 2;
				pCloud[3 * i + var] = (res1[var] + res2[var]) / 2;
			}
		}
	}
}
STRUCTLIGHTCUDA_API void decodeunwarp_gpu(unsigned char *point, int imgnum, int pointnum, double *outputx, double *outputy) {
	//printf("calculate the unwarp\n");
	decodeunwarpkernel <<<GET_BLOCKS(pointnum), CUDA_NUM_THREADS >> > (point, imgnum, pointnum, outputx,outputy);
	//cudaDeviceSynchronize();
}

STRUCTLIGHTCUDA_API void decodewarp_gpu(double *unwarp1, double *unwarp2, double *unwarp3, double *warpped, double wave1, double wave2, double wave3, int pointnum) {
	decode_threefrequencykernel << <GET_BLOCKS(pointnum), CUDA_NUM_THREADS >> >(unwarp1, unwarp2,  unwarp3, warpped, wave1, wave2, wave3,pointnum);
}

STRUCTLIGHTCUDA_API void CorrectPoints_gpu(double *unwarpx, double *unwrapY, int num, int xpixels, int ypixels, double *rpoints)
{
	
	double pixelsPerTwoPiX,pixelsPerTwoPiY;
	pixelsPerTwoPiX = xpixels / 2 / pi;
	pixelsPerTwoPiY = ypixels / 2 / pi;
	createCorrectPoints_rkernel <<<GET_BLOCKS(num), CUDA_NUM_THREADS >> >(unwarpx, unwrapY, num, pixelsPerTwoPiX, pixelsPerTwoPiY, rpoints);
}
STRUCTLIGHTCUDA_API int reconstruct_gpu(double* lPoints, double* rPoints, float *pCloud, int pointsNum
	, float *cpA1, float *cpA2, float *cpAd1, float *cpAd2, float *cpAD) {
	int result=1;
	static int count = 0;
	cudaError_t cudaStatus;

	if (count == 0)
	{
		cudaStatus = cudaMemcpyToSymbol(pA1, cpA1, 9 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 7;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pA2, cpA2, 9 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 8;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd1, cpAd1, 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 9;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd2, cpAd2, 3 * sizeof(float));
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
	reconKernel <<< GET_BLOCKS(pointsNum), CUDA_NUM_THREADS >> >(lPoints, rPoints, pCloud, pointsNum);
Error:
	//释放设备中变量所占内存  
	return result;
}

STRUCTLIGHTCUDA_API double * testadd(double *pg,double *res,int num ) {

	addKernel <<<GET_BLOCKS(num), CUDA_NUM_THREADS >> > (pg, num);
	cudaMemcpy(res, pg, num * sizeof(double), cudaMemcpyDeviceToHost);
	return pg;
}

STRUCTLIGHTCUDA_API int reconstruct3D_gpu(double* lPoints, double* rPoints, float *pCloud, int pointsNum
	, float *cpA1, float *cpA2, float *cpAd1, float *cpAd2, float *cpAD) {
	int result = -1;
	static int count = 0;
	cudaError_t cudaStatus;
	// 从主机内存复制数据到GPU内存中.  
	cudaStatus = cudaMemcpy(dev_pointr, rPoints, 2 * pointsNum * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 6;
		goto Error;
	}
	if (count == 0)
	{
		cudaStatus = cudaMemcpy(dev_pointl, lPoints, 2 * pointsNum * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			result = 6;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pA1, cpA1, 9 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 7;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pA2, cpA2, 9 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 8;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd1, cpAd1, 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			result = 9;
			goto Error;
		}
		cudaStatus = cudaMemcpyToSymbol(pAd2, cpAd2, 3 * sizeof(float));
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


	reconKernel <<< GET_BLOCKS(pointsNum), CUDA_NUM_THREADS >> >(dev_pointl, dev_pointr, dev_p, pointsNum);

	// 采用cudaDeviceSynchronize等待GPU内核函数执行完成并且返回遇到的任何错误信息  
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	result = 7;
	//	goto Error;
	//}

	// 从GPU内存中复制数据到主机内存中  
	cudaStatus = cudaMemcpy(pCloud, dev_p, 3 * pointsNum * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		result = 12;
		goto Error;
	}
	result = 0;
Error:
	//释放设备中变量所占内存  
	return result;
}

STRUCTLIGHTCUDA_API double* doublememcpygpu(int size, double *dirp, double *dp,int model) {
	cudaMemcpy(dirp, dp, size* sizeof(double), cudaMemcpyKind(model));
	return dirp;
}
STRUCTLIGHTCUDA_API float* floatmemcpygpu(int size, float *dirp, float *dp, int model) {
	cudaMemcpy(dirp, dp, size * sizeof(float), cudaMemcpyKind(model));
	return dirp;
}
STRUCTLIGHTCUDA_API unsigned char* ucharmemcpygpu(int size, unsigned char *dirp, unsigned char *dp, int model) {
	cudaMemcpy(dirp, dp, size * sizeof(unsigned char), cudaMemcpyKind(model));
	return dirp;
}


STRUCTLIGHTCUDA_API double* doublememorygpu(int size,double **dp) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)dp,size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		return 0;
	}
	return *dp;
}

STRUCTLIGHTCUDA_API float* floatmemorygpu(int size, float **fp) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)fp, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		return 0;
	}
	return *fp;
}

STRUCTLIGHTCUDA_API unsigned char* ucharmemorygpu(int size, unsigned char **cp) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)cp, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		return 0;
	}
	return *cp;
}
STRUCTLIGHTCUDA_API void  freememorygpu_d(double *p){
	cudaFree(p);
}
STRUCTLIGHTCUDA_API void freememorygpu_f(float *p) {
	cudaFree(p);
}

STRUCTLIGHTCUDA_API void freememorygpu_c(unsigned char *p) {
	cudaFree(p);
}

STRUCTLIGHTCUDA_API int initrecmemorygpu(int size) {
	cudaError_t cudaStatus;
	int result = 0;
	static int inited = 0;
	if (inited == 1)
	{
		return 1;
	}
	// 选择用于运行的GPU  

	//if (opengpu()<0) {
	//	printf("cannot open cuda");
	//	result = 1;
	//}
	// 在GPU中为变量dev_a、dev_b、dev_c分配内存空间.  
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
STRUCTLIGHTCUDA_API int freerecmemorygpu() {
	static int freed = 0;
	if (freed == 1) {
		return 1;
	}
	if (dev_p != 0 || dev_pointr != 0 || dev_pointl != 0)
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
		// 选择用于运行的GPU  
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
		// 选择用于运行的GPU  
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
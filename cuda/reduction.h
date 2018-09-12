#ifndef GPU_REDUCTION_H__
#define GPU_REDUCTION_H__

#include "mathlib.h"
#include "cuarray.h"
#include "rendering.h"
#include <opencv.hpp>

struct MatK {
	float fx, fy, cx, cy;
	MatK() : fx(0), fy(0), cx(0), cy(0) {}
	MatK(float fx_, float fy_, float cx_, float cy_)
	:	fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
	MatK operator()(int pyr) {
		int i = 1 << pyr;
		return MatK(fx / i, fy / i, cx / i, cy / i);
	}
};

void computeVMap(const DeviceArray2D<float> & depth,
				 DeviceArray2D<float4> & vmap,
				 MatK K, float depthCutoff = 3.0f);
void computeVMap(const DeviceArray2D<float4> & vmap,
		         DeviceArray2D<float4> & nmap);
void rgbImageToIntensity(const DeviceArray2D<uchar3> & rgb,
						 DeviceArray2D<unsigned char> & image);
void pyrDownGauss(const DeviceArray2D<float> & src,
		          DeviceArray2D<float> & dst);
void pyrDownGauss(const DeviceArray2D<unsigned char> & src,
		          DeviceArray2D<unsigned char> & dst);
void bilateralFilter(const DeviceArray2D<unsigned short> & depth,
					 DeviceArray2D<float> & filteredDepth,
					 float depthScale);
void icpStep(DeviceArray2D<float4> & nextVMap,
		     DeviceArray2D<float4> & lastVMap,
		     DeviceArray2D<float4> & nextNMap,
		     DeviceArray2D<float4> & lastNMap,
		     Matrix3f Rcurr, float3 tcurr,
		     Matrix3f Rlast, Matrix3f RlastInv,
		     float3 tlast, MatK K,
		     DeviceArray2D<float> & sum,
		     DeviceArray<float> & out,
		     float * residual,
		     double * matrixA_host,
		     double * vectorB_host);
void BuildAdjecencyMatrix(cv::cuda::GpuMat & AM,
						  DeviceArray<ORBKey> & TrainKeys,
						  DeviceArray<ORBKey> & QueryKeys,
						  DeviceArray<float> & MatchDist,
						  DeviceArray<ORBKey> & train_select,
						  DeviceArray<ORBKey> & query_select,
						  DeviceArray<int> & QueryIdx,
						  DeviceArray<int> & SelectedIdx);
#define WarpSize 32
#define MaxThread 1024
#endif

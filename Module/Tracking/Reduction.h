#ifndef GPU_REDUCTION_H__
#define GPU_REDUCTION_H__

#include "VectorMath.h"
#include "Intrinsics.h"
#include "RenderScene.h"

void FilterDepth(const DeviceArray2D<unsigned short> & depth,
		DeviceArray2D<float> & filteredDepth, float depthScale);

void ComputeVMap(const DeviceArray2D<float> & depth,
		DeviceArray2D<float4> & vmap, float fx, float fy, float cx, float cy,
		float depthCutoff);

void ComputeNMap(const DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap);

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);

void PyrDownGauss(const DeviceArray2D<unsigned char> & src,
		DeviceArray2D<unsigned char> & dst);

void ImageToIntensity(const DeviceArray2D<uchar3> & rgb,
		DeviceArray2D<unsigned char> & image);

void ResizeMap(const DeviceArray2D<float4> & vsrc,
		const DeviceArray2D<float4> & nsrc, DeviceArray2D<float4> & vdst,
		DeviceArray2D<float4> & ndst);

void RenderImage(const DeviceArray2D<float4>& points,
		const DeviceArray2D<float4>& normals, const float3 light_pose,
		DeviceArray2D<uchar4>& image);

void DepthToImage(const DeviceArray2D<float> & depth,
		DeviceArray2D<uchar4> & image);

void RgbImageToRgba(const DeviceArray2D<uchar3> & image,
		DeviceArray2D<uchar4> & rgba);

void ICPStep(DeviceArray2D<float4> & nextVMap, DeviceArray2D<float4> & lastVMap,
		DeviceArray2D<float4> & nextNMap, DeviceArray2D<float4> & lastNMap,
		Matrix3f Rcurr, float3 tcurr, Matrix3f Rlast, Matrix3f RlastInv,
		float3 tlast, Intrinsics K, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, float * residual, double * matrixA_host,
		double * vectorB_host);

void SO3Step(const DeviceArray2D<unsigned char> & nextImage,
		const DeviceArray2D<unsigned char> & lastImage, Matrix3f homography,
		Matrix3f kinv, Matrix3f krlr, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, float * redisual, double * matrixA_host,
		double * vectorB_host);

#include <opencv.hpp>
void BuildAdjecencyMatrix(cv::cuda::GpuMat & AM,
		DeviceArray<ORBKey> & TrainKeys, DeviceArray<ORBKey> & QueryKeys,
		DeviceArray<float> & MatchDist, DeviceArray<ORBKey> & train_select,
		DeviceArray<ORBKey> & query_select, DeviceArray<int> & QueryIdx,
		DeviceArray<int> & SelectedIdx);

#endif

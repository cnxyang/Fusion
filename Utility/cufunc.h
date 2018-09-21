#ifndef DEVICE_FUNCTION_HPP__
#define DEVICE_FUNCTION_HPP__

#include "DeviceMap.h"

#include <opencv.hpp>
#include <cuda_runtime.h>

#define WarpSize 32
#define MaxThread 1024

void PyrDownGaussian(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void PyrDownGaussian(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& dst);
void BilateralFiltering(const DeviceArray2D<ushort>& src, DeviceArray2D<float>& dst, float scale);
void ColourImageToIntensity(const DeviceArray2D<uchar3>& src, DeviceArray2D<uchar>& dst);
void ComputeDerivativeImage(const DeviceArray2D<uchar>& src, DeviceArray2D<float>& dx, DeviceArray2D<float>& dy);
void ResizeMap(const DeviceArray2D<float4>& vsrc, const DeviceArray2D<float4>& nsrc, DeviceArray2D<float4>& vdst, DeviceArray2D<float4>& ndst);
void ProjectToDepth(const DeviceArray2D<float4>& src, DeviceArray2D<float>& dst);
void BackProjectPoints(const DeviceArray2D<float>& src, DeviceArray2D<float4>& dst, float depthCutoff, float fx, float fy, float cx, float cy);
void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float4>& dst);
void RenderImage(const DeviceArray2D<float4>& points, const DeviceArray2D<float4>& normals, const float3 light_pose, DeviceArray2D<uchar4>& image);
void forwardProjection(const DeviceArray2D<float4> & vsrc, const DeviceArray2D<float4> & nsrc,  DeviceArray2D<float4> & vdst, DeviceArray2D<float4> & ndst,
		   	   	   	   Matrix3f Rcurr, float3 tcurr, Matrix3f RlastInv, float3 tlast, float fx, float fy, float cx, float cy);
void depthToImage(const DeviceArray2D<float> & depth, DeviceArray2D<uchar4> & image);
void rgbImageToRgba(const DeviceArray2D<uchar3> & image, DeviceArray2D<uchar4> & rgba);

#endif

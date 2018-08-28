#ifndef DEVICE_FUNCTION_HPP__
#define DEVICE_FUNCTION_HPP__

#include "Frame.hpp"
#include "device_array.hpp"
#include "device_map.hpp"

#include <opencv.hpp>
#include <cuda_runtime.h>

#define WarpSize 32
#define MaxThread 1024

void PyrDownGaussian(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void PyrDownGaussian(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& dst);
void BilateralFiltering(const DeviceArray2D<ushort>& src, DeviceArray2D<float>& dst, float scale);
void ColourImageToIntensity(const DeviceArray2D<uchar3>& src, DeviceArray2D<uchar>& dst);
void ComputeDerivativeImage(const DeviceArray2D<uchar>& src, DeviceArray2D<float>& dx, DeviceArray2D<float>& dy);
void ResizeMap(const DeviceArray2D<float4>& vsrc, const DeviceArray2D<float3>& nsrc, DeviceArray2D<float4>& vdst, DeviceArray2D<float3>& ndst);
void ProjectToDepth(const DeviceArray2D<float4>& src, DeviceArray2D<float>& dst);
void BackProjectPoints(const DeviceArray2D<float>& src, DeviceArray2D<float4>& dst, float depthCutoff, float fx, float fy, float cx, float cy);
void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float3>& dst);
void WarpGrayScaleImage(const Frame& frame1, const Frame& frame2, DeviceArray2D<uchar>& diff);
void ComputeResidualImage(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& residual, const Frame& frame);
void RenderImage(const DeviceArray2D<float4>& points, const DeviceArray2D<float3>& normals, const float3 & light_pose, DeviceArray2D<uchar4>& image);

#endif

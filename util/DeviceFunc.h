#ifndef __IMAGE_PROC__
#define __IMAGE_PROC__

#include "Frame.h"
#include "DeviceArray.h"
#include "DeviceStruct.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define WarpSize 32
#define MaxThread 1024

void PyrDownGaussian(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void PyrDownGaussian(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& dst);
void BilateralFiltering(const DeviceArray2D<ushort>& src, DeviceArray2D<float>& dst, float scale);
void ColourImageToIntensity(const DeviceArray2D<uchar3>& src, DeviceArray2D<uchar>& dst);
void ComputeDerivativeImage(const DeviceArray2D<uchar>& src, DeviceArray2D<float>& dx, DeviceArray2D<float>& dy);

void BackProjectPoints(const DeviceArray2D<float>& src, DeviceArray2D<float4>& dst, float depthCutoff, float fx, float fy, float cx, float cy);
void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float3>& dst);
void WarpGrayScaleImage(const Frame& frame1, const Frame& frame2, DeviceArray2D<uchar>& diff);
void ComputeResidualImage(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& residual, const Frame& frame);

void ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int PyrLevel, float* host_a, float* host_b, float& cost);

#endif

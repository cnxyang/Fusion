#ifndef __IMAGE_PROC__
#define __IMAGE_PROC__

#include "Frame.h"
#include "DeviceArray.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void PyrDownGaussian(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void PyrDownGaussian(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& dst);
void BilateralFiltering(const DeviceArray2D<ushort>& src, DeviceArray2D<float>& dst, float scale);
void ColourImageToIntensity(const DeviceArray2D<uchar3>& src, DeviceArray2D<uchar>& dst);

void BackProjectPoints(const DeviceArray2D<float>& src, DeviceArray2D<float4>& dst, float depthCutoff, float fx, float fy, float cx, float cy);
void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float3>& dst);
void WarpGrayScaleImage(const Frame& src1, const Frame& src2, DeviceArray2D<uchar>& diff);

void ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int PyrLevel, float* host_a, float* host_b);

#endif

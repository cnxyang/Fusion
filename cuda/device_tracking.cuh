#ifndef DEVICE_TRACKING_HPP__
#define DEVICE_TRACKING_HPP__

#include "Frame.hpp"

#define WarpSize 32
#define MaxThread 1024

float ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int pyr, float* host_a, float* host_b);
float RGBReduceSum(Frame& NextFrame, Frame& LastFrame,  int pyr, float* host_a, float* host_b);

#endif

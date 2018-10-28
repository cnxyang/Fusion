#pragma once

#include "Utilities/DeviceArray.h"

class Frame;

#define PYRAMID_LEVELS 3

struct PointCloud
{
	PointCloud();

	void importData(Frame* frame, bool useRGB = true);

	bool memoryAllocated;

	DeviceArray2D<uchar> image[PYRAMID_LEVELS];
	DeviceArray2D<float> depth[PYRAMID_LEVELS];
	DeviceArray2D<short> dIdx[PYRAMID_LEVELS];
	DeviceArray2D<short> dIdy[PYRAMID_LEVELS];
	DeviceArray2D<float4> vmap[PYRAMID_LEVELS];
	DeviceArray2D<float4> nmap[PYRAMID_LEVELS];
	DeviceArray2D<uchar3> image_raw;
	DeviceArray2D<float> depth_raw;

	Frame * frame;
};

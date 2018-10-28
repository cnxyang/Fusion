#pragma once

#include "Utilities/DeviceArray.h"

class Frame;

#define PYRAMID_LEVELS 3

struct PointCloud
{
	PointCloud(): memoryAllocated(false) {}

	bool memoryAllocated;

	DeviceArray2D<float4> points;
	DeviceArray2D<float4> normal;
	DeviceArray2D<uchar3> rawColor;
	DeviceArray2D<ushort> rawDepth;
	DeviceArray2D<uchar> image;
	DeviceArray2D<float> depth;
	DeviceArray2D<short> dIdx;
	DeviceArray2D<short> dIdy;
};

class TrackingReference
{
public:

	TrackingReference(int trackingLevel);

	void populateICPData(Frame * source, bool useRGB = true);

	PointCloud cloud[PYRAMID_LEVELS];

	Frame * frame;
};

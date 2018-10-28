#pragma once

#include "Utilities/DeviceArray.h"

class Frame;

struct PointCloud
{
	DeviceArray2D<float4> points;
	DeviceArray2D<float4> normal;
	DeviceArray2D<unsigned short> rawDepth;
	DeviceArray2D<unsigned char> image;
	DeviceArray2D<float> depth;
	DeviceArray2D<short> dIdx, dIdy;
};

class TrackingReference
{
public:

	TrackingReference(int trackingLevel);

	void populateData(Frame* source, bool useRGB = true);

	PointCloud* cloud;
};

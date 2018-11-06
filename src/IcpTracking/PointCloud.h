#pragma once

#include "Utilities/DeviceArray.h"

class Frame;

struct PointCloud
{
	static const int NUM_PYRS = 3;
	PointCloud();
	PointCloud(const PointCloud&) = delete;
	PointCloud& operator=(const PointCloud&) = delete;

	~PointCloud();

	void generateCloud(Frame* frame, bool useRGB = true);
	void setReferenceFrame(Frame* frame);
	void generatePyramid();

	bool memoryAllocated;

	DeviceArray2D<uchar> image[NUM_PYRS];
	DeviceArray2D<float> depth[NUM_PYRS];
	DeviceArray2D<short> dIdx[NUM_PYRS];
	DeviceArray2D<short> dIdy[NUM_PYRS];
	DeviceArray2D<float4> vmap[NUM_PYRS];
	DeviceArray2D<float4> nmap[NUM_PYRS];
	DeviceArray2D<uchar3> image_raw;
	DeviceArray2D<float> depth_float;
	DeviceArray2D<unsigned short> depth_ushort;

	Frame * frame;
};

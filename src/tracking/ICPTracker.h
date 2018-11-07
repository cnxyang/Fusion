#pragma once

#include "DeviceArray.h"
#include "SophusUtil.h"
#include <Eigen/Core>

class Frame;
class PointCloud;

class ICPTracker
{
public:

	static const int NUM_PYRS = 3;

	ICPTracker(int w, int h, Eigen::Matrix3f K);
	~ICPTracker();

	SE3 trackSE3(PointCloud* ref, PointCloud* target, SE3 initValue = SE3(), bool useRGB = true);

	bool trackingWasGood;
	float lastIcpError;
	float lastRgbError;
	float icpInlierRatio;
	float rgbInlierRatio;

	inline void setIterations(std::vector<float> iter);

protected:

	// temporary variables
	// used for ICP reduction
	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSE3;
	DeviceArray<int> outRES;
	DeviceArray2D<int> sumRES;
	const float RGBWeight = 0.0001f;

	// the number of iterations per layer
	// NOTE: should set manually before tracking
	int iterations[NUM_PYRS];
};

inline void ICPTracker::setIterations(std::vector<float> iter)
{
	for (int level = 0; level < NUM_PYRS; ++level)
		iterations[level] = iter[level];
}

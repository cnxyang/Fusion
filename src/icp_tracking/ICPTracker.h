#pragma once

#include "DeviceArray.h"
#include "DeviceFuncs.h"
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
	inline Eigen::Matrix<double, 6, 6> getInformation() const;

protected:

	// temporary variables
	// used for ICP reduction
	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSE3;
	DeviceArray<int> outRES;
	DeviceArray2D<int> sumRES;
	DeviceArray<float> out;
	DeviceArray2D<float> sum;
	DeviceArray2D<CorrespItem> corresp_image;

	const float RGBWeight = 0.0001f;

	// Total Reduction results
	Eigen::Matrix<double, 6, 6> matrixA;
	Eigen::Matrix<double, 6, 1> vectorb;

	// ICP reduction results
	Eigen::Matrix<double, 6, 6> matrixAicp;
	Eigen::Matrix<double, 6, 1> vectorbicp;

	// RGB reduction results
	Eigen::Matrix<double, 6, 6> matrixArgb;
	Eigen::Matrix<double, 6, 1> vectorbrgb;

	// se3
	Eigen::Matrix<double, 6, 1> result;

	// the number of iterations per layer
	// NOTE: should set manually before tracking
	int iterations[NUM_PYRS];
};

inline void ICPTracker::setIterations(std::vector<float> iter)
{
	for (int level = 0; level < NUM_PYRS; ++level)
		iterations[level] = iter[level];
}
inline Eigen::Matrix<double, 6, 6> ICPTracker::getInformation() const
{
	return matrixA.inverse();
}

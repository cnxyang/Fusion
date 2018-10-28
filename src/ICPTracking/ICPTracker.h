#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "Utilities/DeviceArray.h"
#include "Utilities/SophusUtil.h"

class Frame;
class PointCloud;

#define PYRAMID_LEVELS 3

class ICPTracker
{
public:
	ICPTracker(int w, int h, Eigen::Matrix3f K);

	~ICPTracker();

	inline void setIterations(int * iterations);

	inline void setTrackingLevel(int begin = 2, int end = 0);

	Sophus::SE3d trackSE3(Frame* trackingReference,
			Frame* trackingTarget,
			Sophus::SE3d frameToRef_initialEstimate = SE3(),
			bool addImageAlignment = true);

	Sophus::SE3d trackSE3(PointCloud* trackingReference,
			Frame* trackingTarget,
			Sophus::SE3d frameToRef_initialEstimate = SE3(),
			bool addImageAlignment = true);

	Sophus::SE3d trackSE3(PointCloud* trackingReference,
			PointCloud* trackingTarget,
			Sophus::SE3d frameToRef_initialEstimate = SE3(),
			bool addImageAlignment = true);

	bool trackingWasGood;
	float lastIcpError;
	float lastRgbResidual;
	float icpInlierRatio;
	float rgbInlierRatio;

protected:

	PointCloud * trackingReference;
	PointCloud * trackingTarget;

	// used for ICP reduction
	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSE3;

	int iterations[PYRAMID_LEVELS];

	int trackingLevelBegin;
	int trackingLevelEnd;
};

inline void ICPTracker::setIterations(int * iter)
{
	for(int level = 0; level < PYRAMID_LEVELS; ++level)
		iterations[level] = iter[level];
}

inline void ICPTracker::setTrackingLevel(int begin, int end)
{
	trackingLevelBegin = begin;
	trackingLevelEnd = end;
}

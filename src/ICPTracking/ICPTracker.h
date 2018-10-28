#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

class Frame;
class TrackingReference;

class ICPTracker
{
public:
	ICPTracker(int w, int h, Eigen::Matrix3f K);

	~ICPTracker();

	inline void setTrackingLevel(int begin = 0, int end = 2);

	void initTrackingData();

	Sophus::SE3d trackFrame(Frame* reference, Frame* newFrame, Sophus::SE3d r2f_initialGuess, bool useRGB = true);

	Sophus::SE3d trackFrame(TrackingReference* reference, Frame* frame, Sophus::SE3d r2f_initialGuess, bool useRGB = true);

private:

	TrackingReference* data;

	int trackingLevelBegin, trackingLevelEnd;
};

inline void ICPTracker::setTrackingLevel(int begin, int end)
{
	trackingLevelBegin = begin;
	trackingLevelEnd = end;
}

#include "ICPTracker.h"
#include "DeviceFuncs.h"
#include "TrackingReference.h"

ICPTracker::ICPTracker(int w, int h, Eigen::Matrix3f K) :
	trackingLevelBegin(0), trackingLevelEnd(0), data(0)
{

}

ICPTracker::~ICPTracker()
{
	delete data;
}

void ICPTracker::initTrackingData()
{
	data = new TrackingReference(trackingLevelEnd - trackingLevelBegin + 1);
}

Sophus::SE3d ICPTracker::trackFrame(
		TrackingReference * reference,
		Frame* frame,
		Sophus::SE3d initialEstimate,
		bool useRGB)
{

}

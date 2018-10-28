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
		TrackingReference* reference,
		Frame* frame,
		Sophus::SE3d r2f_initialGuess,
		bool useRGB)
{
//	data->populateData(frame, useRGB);
//	float icpResidual[2];
//
//	for(int iter = trackingLevelBegin; iter < trackingLevelEnd; ++iter)
//	{
//		ICPStep(data->cloud[iter].point, reference->cloud[iter].point,
//				data->cloud[iter].normal, reference->cloud[iter].normal);
//
//		if(useRGB)
//		{
//			RGBStep();
//		}
//	}
}

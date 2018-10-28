#include "ICPTracking/Relocaliser.h"
#include "ICPTracking/ICPTracker.h"

Relocaliser::Relocaliser(int w, int h, Eigen::Matrix3f K) :
	maxNumAttempts(0), relocaliseTracker(0)
{
	relocaliseTracker = new ICPTracker(w, h, K);
	relocaliseTracker->setTrackingLevel(0, 1);
	relocaliseTracker->initTrackingData();
}

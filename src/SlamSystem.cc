#include "SlamSystem.h"
#include "DataStructure/Frame.h"
#include "GlobalMapping/DenseMap.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "ICPTracking/Relocaliser.h"
#include "ICPTracking/ICPTracker.h"

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled) :
	width(w), height(h), camK(K), tracker(0),
	constraintTracker(0), map(0), keyFrameGraph(0),
	currentKeyFrame(0), lastTrackingScore(0), keepRunning(true),
	SLAMEnabled(SLAMEnabled), relocaliser(0), dumpMapToDisk(false),
	trackingIsGood(true), trackingReference(0)
{
	map = new DenseMap();

	tracker = new ICPTracker(width, height, camK);
	tracker->setTrackingLevel(1, 3);
	tracker->initTrackingData();

	constraintTracker = new ICPTracker(width, height, camK);
	constraintTracker->setTrackingLevel(1, 2);
	constraintTracker->initTrackingData();

	threadConstraintSearch = std::thread(&SlamSystem::constraintSearchLoop, this);
	threadOptimisation = std::thread(&SlamSystem::optimisationLoop, this);
	threadMapping = std::thread(&SlamSystem::mappingLoop, this);
}

void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& dp, double timeStamp)
{
//	std::shared_ptr<Frame*> newFrame(new Frame());
//
//	if(!trackingIsGood)
//	{
//		relocaliser.updateCurrentFrame(newFrame);
//		return;
//	}
//
//	currentKeyFrameMutex.lock();
//
//	Sophus::SE3d newRefToFrame_poseUpdate = tracker->trackFrame(
//			referenceFrame, newFrame,
//			frameToReference_initialEstimate);
//
//	lastResidual = tracker->lastResidual;
//	lastUsage = tracker->pointUsage;
//
//	if(tracker->diverged || !tracker->trackingWasGood)
//	{
//		printf("LOST TRACKING for frame %d\n", newFrame->frameId);
//		trackingIsGood = false;
//		nextRelocId = -1;
//
//		return;
//	}
//
//	keyFrameGraph->addFrame(newFrame.get());
//
//	lastTrackedFrame = trackingNewFrame;
}

void SlamSystem::populateICPData()
{

}

void SlamSystem::mappingLoop()
{
	printf("mapping thread started\n");

	while(keepRunning)
	{
		if(dumpMapToDisk)
		{

		}
	}
}

void SlamSystem::optimisationLoop()
{
	printf("optimisation thread started\n");

	while(keepRunning)
	{

	}

	printf("optimisation thread exited\n");
}

void SlamSystem::constraintSearchLoop()
{
	printf("constraint searching thread started\n");
	std::unique_lock<std::mutex> lock(newKeyFrameMutex);

	while(keepRunning)
	{
		if(newKeyFrames.size() == 0)
		{
			lock.unlock();
		}
		else
		{
			Frame* newKF = newKeyFrames.front();
			newKeyFrames.pop_front();
			lock.unlock();

			findConstraintForNewKF(newKF);

			lock.lock();
		}
	}

	printf("constraint searching thread exited\n");
}

void SlamSystem::findConstraintForNewKF(Frame* newKF)
{

}

#include "SlamSystem.h"
#include "DataStructure/Frame.h"
#include "ICPTracking/TrackingReference.h"
#include "GlobalMapping/DenseMapping.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "ICPTracking/ICPTracker.h"

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled) :
	width(w), height(h), K(K), tracker(0), constraintTracker(0), map(0), keyFrameGraph(0),
	currentKeyFrame(0), lastTrackingScore(0), SLAMEnabled(SLAMEnabled), dumpMapToDisk(false),
	trackingIsGood(true), trackingReference(0), keepRunning(true), currentFrame(0),
	viewer(0), lastTrackedFrame(0)
{
	map = new DenseMapping();

	viewer = new SlamViewer();

	tracker = new ICPTracker(width, height, K);
	tracker->setTrackingLevel(1, 3);
	tracker->initTrackingData();

	constraintTracker = new ICPTracker(width, height, K);
	constraintTracker->setTrackingLevel(1, 2);
	constraintTracker->initTrackingData();

	thread_constraintSearch = std::thread(&SlamSystem::constraintSearchLoop, this);
	thread_optimisation = std::thread(&SlamSystem::optimisationLoop, this);
	thread_mapping = std::thread(&SlamSystem::mappingLoop, this);
	thread_visualisation = std::thread(&SlamViewer::spin, viewer);
}

void SlamSystem::trackFrame(cv::Mat & image, cv::Mat & depth, int id, double timeStamp)
{
	currentFrame = new Frame(image, depth, id, K, timeStamp);

	if (id == 0 && !trackingReference)
	{
		trackingReference = new TrackingReference(3);
		trackingReference->populateICPData(currentFrame, true);
		return;
	}

	if (!trackingIsGood)
	{
		printf("TRACKING FAILED for Frame %d.\n", id - 1);
		printf("WARNING: Relocalization NOT Implemented at the moment.\n");
		return;
	}

	currentKeyFrameMutex.lock();

//	Sophus::SE3d initialEstimate;
//	if(lastTrackedFrame)
//		initialEstimate = lastTrackedFrame->getCamToWorld() * currentKeyFrame->getCamToWorld().inverse();

//	Sophus::SE3d camToRef = tracker->trackFrame(
//			trackingReference,
//			currentFrame,
//			initialEstimate,
//			true);

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

//	keyFrameGraph->addFrame(currentFrame);

	lastTrackedFrame = currentFrame;
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

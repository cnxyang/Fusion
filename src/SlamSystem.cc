#include "SlamSystem.h"
#include "DataStructure/Frame.h"
#include "ICPTracking/PointCloud.h"
#include "GlobalMapping/DistanceField.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "ICPTracking/ICPTracker.h"

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled) :
	width(w), height(h), K(K), tracker(0), constraintTracker(0), map(0), keyFrameGraph(0),
	currentKeyFrame(0), lastTrackingScore(0), SLAMEnabled(SLAMEnabled), dumpMapToDisk(false),
	trackingIsGood(true), trackingReference(0), keepRunning(true), trackingNewFrame(0),
	viewer(0), lastTrackedFrame(0), newConstraintAdded(false), frameToMapTracking(true)
{
	map = new DistanceField();
	map->allocateDeviceMemory();

	viewer = new SlamViewer();
	viewer->setMap(map);

	// main tracking thread
	tracker = new ICPTracker(w, h, K);

	// ICP based tracking for searching constraints of new key frames
	constraintTracker = new ICPTracker(w, h, K);
	constraintTracker->setTrackingLevel(2, 1);

	// start multi-threading!
	threadConstraintSearch = std::thread(&SlamSystem::constraintSearchLoop, this);
	threadOptimisation = std::thread(&SlamSystem::optimisationLoop, this);
	threadVisualisation = std::thread(&SlamSystem::visualisationLoop, this);
	threadDenseMapping = std::thread(&SlamSystem::mappingLoop, this);

	keyFrameGraph = new KeyFrameGraph();

	trackingReference = new PointCloud();
}

SlamSystem::~SlamSystem()
{
	keepRunning = false;

	printf("Waiting for other threads to quit.\n");
	newKeyFrameCreatedSignal.notify_all();
	threadDenseMapping.join();
	threadOptimisation.join();
	threadVisualisation.join();
	threadConstraintSearch.join();
	printf("DONE waiting.\n");

	delete map;
	delete tracker;
	delete constraintTracker;
	delete keyFrameGraph;
	delete viewer;
}

void SlamSystem::trackFrame(cv::Mat & image, cv::Mat & depth, int id, double timeStamp)
{
	trackingNewFrame = new Frame(image, depth, id, K, timeStamp);

	if (!trackingIsGood)
	{
		printf("WARNING: Re-localisation NOT Implemented at the moment.\n");
		return;
	}

	currentKeyFrameMutex.lock();
	if(trackingReference->frame != currentKeyFrame)
	{
		if(frameToMapTracking && currentKeyFrame->id() > 0)
		{

		}
		else
		{
			trackingReference->importFrame(trackingNewFrame, true);
		}
	}

	FramePoseStruct* trackingReferencePose = trackingReference->frame->poseStruct;
	currentKeyFrameMutex.unlock();

	SE3 frameToRef_initialEstimate =
			currentKeyFrame->getCamToWorld().inverse() *
			lastTrackedFrame->getCamToWorld();

	SE3 frameToRef = tracker->trackSE3(
			trackingReference,
			trackingNewFrame,
			frameToRef_initialEstimate);

	if(!tracker->trackingWasGood)
	{
		printf("TRACKING LOST for frame %d.\n", trackingNewFrame->id());
	}
}

void SlamSystem::visualisationLoop()
{
	printf("visualisation thread started\n");

	while(keepRunning)
	{
		int i = 0;
	}

	printf("visualisation thread exited\n");
}

void SlamSystem::mappingLoop()
{
	printf("mapping thread started\n");

	while(keepRunning)
	{
		if(dumpMapToDisk)
		{
			int i = 0;
		}
	}

	printf("mapping thread exited\n");
}

void SlamSystem::optimisationLoop()
{
	printf("optimisation thread started\n");

	while(keepRunning)
	{
		std::unique_lock<std::mutex> lock(newConstraintMutex);

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
		lock.lock();
	}

	printf("constraint searching thread exited\n");
}

void SlamSystem::findConstraintForNewKF(Frame* newKF)
{

}

void SlamSystem::createNewKeyFrame(Frame* candidateKF)
{

}

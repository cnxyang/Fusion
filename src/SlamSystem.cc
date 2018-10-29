#include "SlamSystem.h"
#include "DataStructure/Frame.h"
#include "ICPTracking/PointCloud.h"
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
	if(viewer != 0) delete viewer;
}

void SlamSystem::trackFrame(cv::Mat & image, cv::Mat & depth, int id, double timeStamp)
{
	currentFrame = new Frame(image, depth, id, K, timeStamp);

	if (id == 0 && !trackingReference)
	{
		trackingReference = new PointCloud();
		trackingReference->importData(currentFrame, true);
		return;
	}

	SE3 frameToRef_initialEstimate = lastTrackedFrame->pose.inverse() * currentKeyFrame->pose;
	SE3 frameToRef = tracker->trackSE3(trackingReference, currentFrame, frameToRef_initialEstimate);

	if (!trackingIsGood)
	{
		printf("TRACKING FAILED for Frame %d.\n", id - 1);
		printf("WARNING: Relocalization NOT Implemented at the moment.\n");
		return;
	}

	currentFrame->pose = currentKeyFrame->pose * frameToRef.inverse();

	lastTrackedFrame = currentFrame;

	viewer->setCurrentCamPose(lastTrackedFrame->getCamToWorld());
}

void SlamSystem::visualisationLoop()
{
	printf("visualisation thread started\n");

	while(keepRunning)
	{
		viewer->run();
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

		}
	}

	printf("mapping thread exited\n");
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

void SlamSystem::createNewKeyFrame(Frame* newKeyFrameCandidate)
{
	currentKeyFrameMutex.lock();
	currentKeyFrame = newKeyFrameCandidate;
	currentKeyFrameMutex.unlock();
}

#pragma once

#include <Eigen/Core>
#include <opencv.hpp>
#include <thread>
#include <condition_variable>

class Frame;
class SlamViewer;
class ICPTracker;
class PointCloud;
class KeyFrameGraph;
class DistanceField;

class SlamSystem
{
public:

	SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled = true);
	SlamSystem(const SlamSystem&) = delete;
	SlamSystem& operator=(const SlamSystem&) = delete;
	~SlamSystem();

	void trackFrame(cv::Mat & image, cv::Mat & depth, int id, double time_stamp);

protected:

	void mappingLoop();
	void constraintSearchLoop();
	void optimisationLoop();
	void visualisationLoop();
	void findConstraintForNewKF(Frame* newKeyFrame);
	void createNewKeyFrame(Frame* candidateFrame);

	void rebootSlamSystem();
	void renderBirdsEyeView();

	int width, height;
	Eigen::Matrix3f K;

	SlamViewer * viewer;
	DistanceField * map;
	ICPTracker * tracker;
	ICPTracker * constraintTracker;
	KeyFrameGraph * keyFrameGraph;

	bool keepRunning;
	bool SLAMEnabled;
	bool dumpMapToDisk;

	// multi-threading variables
	std::thread threadConstraintSearch;
	std::thread threadOptimisation;
	std::thread threadDenseMapping;
	std::thread threadVisualisation;

	// PUSHED by tracking, READ && CLEARED BY constraint finder
	std::deque<Frame *> newKeyFrames;
	std::mutex newKeyFrameMutex;
	std::condition_variable newKeyFrameCreatedSignal;

	// PROCESSED by tracking && pose graph
	Frame * currentKeyFrame;
	std::mutex currentKeyFrameMutex;

	bool trackingIsGood;
	float lastTrackingScore;
	PointCloud * trackingReference;
	Frame * trackingNewFrame;
	Frame * lastTrackedFrame;
	bool frameToMapTracking;

	// optimization thread
	bool newConstraintAdded;
	std::mutex newConstraintMutex;
	std::condition_variable newConstraintCreatedSignal;
	std::mutex g2oGraphAccessMutex;
};

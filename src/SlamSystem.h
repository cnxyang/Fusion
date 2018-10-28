#pragma once

#include <Eigen/Core>
#include <opencv.hpp>
#include <thread>
#include <condition_variable>

class Frame;
class DenseMapping;
class SlamViewer;
class ICPTracker;
class KeyFrameGraph;
class TrackingReference;

class SlamSystem
{
public:

	SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled = true);

	void initSystem();
	void trackFrame(cv::Mat & image, cv::Mat & depth, int id, double timeStamp);

protected:

	void mappingLoop();
	void constraintSearchLoop();
	void optimisationLoop();
	void findConstraintForNewKF(Frame* newKF);
	void populateICPData();

	int width, height;
	Eigen::Matrix3f K;

	SlamViewer * viewer;
	DenseMapping * map;
	ICPTracker * tracker;
	ICPTracker * constraintTracker;
	KeyFrameGraph * keyFrameGraph;

	float lastTrackingScore;
	bool keepRunning, SLAMEnabled;
	bool dumpMapToDisk;

	std::thread thread_constraintSearch;
	std::thread thread_optimisation;
	std::thread thread_mapping;
	std::thread thread_visualisation;

	// PUSHED by tracking, READ && CLEARED BY constraint finder
	std::deque<Frame*> newKeyFrames;
	std::mutex newKeyFrameMutex;
	std::condition_variable newKeyFrameCreatedSignal;

	// PROCESSED by tracking && pose graph
	Frame* currentFrame;
	Frame* currentKeyFrame;
	std::mutex currentKeyFrameMutex;

	bool trackingIsGood;
	TrackingReference* trackingReference; // for current key frame
	Frame * lastTrackedFrame;
};

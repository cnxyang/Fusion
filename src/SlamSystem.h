#pragma once

#include <thread>
#include <Eigen/Core>
#include <opencv.hpp>
#include <condition_variable>

class Frame;
class DenseMap;
class MapViewer;
class ICPTracker;
class KeyFrameGraph;
class Relocaliser;
class TrackingReference;

class SlamSystem
{
public:

	SlamSystem(int w, int h, Eigen::Matrix3f K, bool SLAMEnabled = true);

	void initSystem();
	void trackFrame(cv::Mat& img, cv::Mat& dp, double timeStamp);

protected:

	void mappingLoop();
	void constraintSearchLoop();
	void optimisationLoop();
	void findConstraintForNewKF(Frame* newKF);
	void populateICPData();

	int width, height;
	Eigen::Matrix3f camK;

	DenseMap* map;
	ICPTracker* tracker;
	ICPTracker* constraintTracker;
	KeyFrameGraph* keyFrameGraph;
	Relocaliser* relocaliser;

	float lastTrackingScore;
	bool keepRunning, SLAMEnabled;
	bool dumpMapToDisk;

	std::thread threadConstraintSearch;
	std::thread threadOptimisation;
	std::thread threadMapping;

	// PUSHED by tracking, READ && CLEARED BY constraint finder
	std::deque<Frame*> newKeyFrames;
	std::mutex newKeyFrameMutex;
	std::condition_variable newKeyFrameCreatedSignal;

	// PROCESSED by tracking && pose graph
	Frame* currentKeyFrame;
	std::mutex currentKeyFrameMutex;

	bool trackingIsGood;
	TrackingReference* trackingReference; // for current key frame
};

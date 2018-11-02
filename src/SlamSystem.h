#pragma once

#include <mutex>
#include <thread>
#include <atomic>
#include <Eigen/Core>
#include <opencv.hpp>
#include "DeviceArray.h"

class Frame;
class GlViewer;
class Tracker;
class VoxelMap;
class PointCloud;
class ICPTracker;
class KeyFrameGraph;

// Msg is used for communicating
// with the system, mainly from
// the visualisation thread.
struct Msg
{
	Msg(int msg) : data(msg) {}

	enum
	{
		EMPTY_MSG,
		SYSTEM_RESET,
		EXPORT_MESH_TO_FILE,
		WRITE_BINARY_MAP_TO_DISK,
		READ_BINARY_MAP_FROM_DISK,
		SYSTEM_SHUTDOWN,
		TOGGLE_MESH_ON,
		TOGGLE_MESH_OFF,
		TOGGLE_IMAGE_ON,
		TOGGLE_IMAGE_OFF,
	};

	int data;
};

class SlamSystem
{
public:

	SlamSystem(int w, int h, Eigen::Matrix3f K);

	SlamSystem(const SlamSystem&) = delete;
	SlamSystem& operator=(const SlamSystem&) = delete;

	~SlamSystem();

	// Public APIs
	void trackFrame(cv::Mat& image, cv::Mat& depth, int id, double timeStamp);

	bool shouldQuit() const;
	void queueMessage(Msg newmsg);


protected:

	// Message loop
	void processMessages();

	// Utils
	void rebootSystem();
	void exportMeshAsFile();
	void systemReInitialise();
	void writeBinaryMapToDisk();
	void readBinaryMapFromDisk();

	// Try build pose graph
	void updateVisualisation();
	void findConstraintsForNewKeyFrames(Frame* newKF);
	void checkConstraints();
	void tryTrackConstraint();

	// Sub-routines
	VoxelMap* map;
	GlViewer* viewer;

	// General control variables
	bool keepRunning;
	bool systemRunning;

	// Camera intrinsics
	Eigen::Matrix3f K;

	// Image parameters
	int width, height;

	// Multi-threading loop
	void loopVisualisation();
	void loopOptimization();
	void loopConstraintSearch();

	// Multi-threading variables
	std::thread threadVisualisation;
	std::thread threadOptimization;
	std::thread threadConstraintSearch;

	Frame* currentKeyFrame;
	Frame* latestTrackedFrame;

	KeyFrameGraph* keyFrameGraph;

	// Used for frame-to-model tracking
	PointCloud* trackingReference;
	PointCloud* trackingTarget;
	ICPTracker* tracker;
	ICPTracker* constraintTracker;

	// Used for constraint searching
	std::deque<Frame*> newKeyFrames;
	std::mutex newKeyFrameMutex;

	std::mutex messageQueueMutex;
	std::queue<Msg> messageQueue;

	// Images used for debugging
	void displayDebugImages(int ms);

	cv::Mat imageReference;
	cv::Mat depthReference;
	cv::Mat imageTarget;
	cv::Mat depthTarget;
	cv::Mat nmapReference;
	cv::Mat nmapTarget;
};

inline bool SlamSystem::shouldQuit() const
{
	return !systemRunning;
}

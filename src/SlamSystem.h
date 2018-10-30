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
class DenseMap;
class PointCloud;
class ICPTracker;
class KeyFrameGraph;

class SlamSystem
{
public:

	void RebootSystem();

	void processMessages(bool finished = false);

	void WriteMeshToDisk();

	void WriteMapToDisk();

	void ReadMapFromDisk();

	void renderBirdsEyeView(float dist = 8.0f);

	std::atomic<bool> paused;
	std::atomic<bool> requestMesh;
	std::atomic<bool> requestSaveMap;
	std::atomic<bool> requestReadMap;
	std::atomic<bool> requestSaveMesh;
	std::atomic<bool> requestReboot;
	std::atomic<bool> requestStop;
	std::atomic<bool> imageUpdated;
	std::atomic<bool> poseOptimized;

	DeviceArray2D<float4> vmap;
	DeviceArray2D<float4> nmap;
	DeviceArray2D<uchar4> renderedImage;

	cv::Mat mK;

	void ReIntegration();

	Tracker* tracker;
	GlViewer* viewer;
	DenseMap* map;

//=================== REFACTORING ====================

	SlamSystem(int w, int h, Eigen::Matrix3f K);
	SlamSystem(const SlamSystem&) = delete;
	SlamSystem& operator=(const SlamSystem&) = delete;
	~SlamSystem();

	// Public APIs
	void rebootSystem();
	bool shouldQuit() const;
	void trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp);

private:

	void systemReInitialise();
	void updateVisualisation();
	void findConstraintsForNewKeyFrames(Frame* newKF);

	// General control variables
	bool keepRunning;
	bool systemRunning;
	bool dumpMapToDisk;

	// Camera intrinsics
	Eigen::Matrix3f K;

	// Image parameters
	int width, height;
	int numImagesProcessed;

	// Multi-threading loop
	void loopVisualisation();
	void loopMapUpdate();
	void loopOptimization();
	void loopConstraintSearch();

	// Multi-threading variables
	std::thread threadVisualisation;
	std::thread threadOptimization;
	std::thread threadMapUpdate;
	std::thread threadConstraintSearch;

	Frame* currentKeyFrame;
	Frame* latestTrackedFrame;

	KeyFrameGraph* keyFrameGraph;

	// Used for frame-to-model tracking
	PointCloud* trackingReference;
	PointCloud* trackingTarget;
	ICPTracker* mainTracker;

	// Used for constraint searching
	std::deque<Frame*> newKeyFrames;
	std::mutex newKeyFrameMutex;

	struct Mesh
	{
		DeviceArray<float4> vertex;
		DeviceArray<float4> normal;
		DeviceArray<uchar4> color;
	} mesh;
};

inline bool SlamSystem::shouldQuit() const
{
	return !systemRunning;
}

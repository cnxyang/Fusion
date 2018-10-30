#ifndef SYSTEM_H__
#define SYSTEM_H__

#include "Tracking.h"
#include "SlamViewer.h"
#include "Optimizer.h"
#include "DistanceField.h"

#include <thread>

class SlamViewer;
class Tracker;
class DistanceField;
class Optimizer;

struct SysDesc {
	int cols, rows;
	float fx;
	float fy;
	float cx;
	float cy;
	float DepthCutoff;
	float DepthScale;
	bool TrackModel;
	std::string path;
	bool bUseDataset;
};

class System
{
public:



	System(SysDesc * pParam);


	void JoinViewer();

	void RebootSystem();

	void FilterMessage(bool finished = false);

	void WriteMeshToDisk();

	void WriteMapToDisk();

	void ReadMapFromDisk();

	void RenderTopDown(float dist = 8.0f);

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
	int nFrames;
	bool state;

	void ReIntegration();

	DistanceField * map;
	SysDesc * param;
	SlamViewer  * viewer;
	Tracker * tracker;
	Optimizer * optimizer;

	std::thread * viewerThread;
	std::thread * optimizerThd;

	int num_frames_after_reloc;

	// =============== REFACTORING ====================

	System(int w, int h, Eigen::Matrix3f K);
	System(const System&) = delete;
	System& operator=(const System&) = delete;

	bool trackFrame(cv::Mat& image, cv::Mat& depth, int id, double timeStamp);

	Frame* currentKeyFrame;
	Frame* trackingNewFrame;

	Eigen::Matrix3f K;
	int imageWidth, imageHeight;
};

#endif

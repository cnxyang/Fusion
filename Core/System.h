#ifndef SYSTEM_HPP__
#define SYSTEM_HPP__

#include "Viewer.h"
#include "Mapping.h"
#include "Tracking.h"

#include <mutex>
#include <thread>
#include <future>

class Viewer;
class Mapping;
class Tracker;

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

class System {
public:
	System(const char* str);
	System(SysDesc* pParam);
	bool grabImage(const cv::Mat& image, const cv::Mat& depth);
	void joinViewer();
	void saveMesh();
	void reboot();

public:
	Mapping * mpMap;
	Viewer * mpViewer;
	SysDesc * param;
	Tracker * mpTracker;

	cv::Mat mK;
	int nFrames;
	std::thread * mptViewer;

	bool state;
	cudaStream_t stream;
	std::mutex mutexReq;
	std::atomic<bool> paused;
	std::atomic<bool> requestMesh;
	std::atomic<bool> requestSaveMesh;
	std::atomic<bool> requestReboot;
	std::atomic<bool> requestStop;
	std::atomic<bool> imageUpdated;
	DeviceArray2D<float4> vmap;
	DeviceArray2D<float4> nmap;
	DeviceArray2D<uchar4> renderedImage;
};

#endif

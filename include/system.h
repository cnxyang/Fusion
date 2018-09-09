#ifndef SYSTEM_HPP__
#define SYSTEM_HPP__

#include "map.h"
#include "viewer.h"
#include "tracker.h"

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
	std::mutex mutexReq;
	std::atomic<bool> paused;
	std::atomic<bool> requestSaveMesh;
	std::atomic<bool> requestReboot;
	std::atomic<bool> requestStop;
};

#endif

#ifndef SYSTEM_HPP__
#define SYSTEM_HPP__

#include "Mapping.hpp"
#include "Viewer.hpp"
#include "Tracking.hpp"

#include <mutex>
#include <thread>

class Viewer;
class Mapping;
class tracker;

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
	void GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);
	void SetParameters(SysDesc& desc);
	void PrintTimings();
	void JoinViewer();
	void saveMesh();
	void Reboot();
	void Stop();

public:
	Mapping* mpMap;
	Viewer* mpViewer;
	SysDesc* mpParam;
	tracker* mpTracker;

	cv::Mat mK;
	bool mbStop;
	int nFrames;
	std::thread* mptViewer;

	std::mutex mutexReq;
	bool requestSaveMesh;
};

#endif

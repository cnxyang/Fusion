#ifndef SYSTEM_H__
#define SYSTEM_H__

#include "Viewer.h"
#include "Mapping.h"
#include "Tracking.h"

#include <thread>

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

	System(SysDesc * pParam);

	bool GrabImage(const cv::Mat & image, const cv::Mat & depth);

	void JoinViewer();

	void SaveMesh();

	void Reboot();

	std::atomic<bool> paused;
	std::atomic<bool> requestMesh;
	std::atomic<bool> requestSaveMesh;
	std::atomic<bool> requestReboot;
	std::atomic<bool> requestStop;
	std::atomic<bool> imageUpdated;
	DeviceArray2D<float4> vmap;
	DeviceArray2D<float4> nmap;
	DeviceArray2D<uchar4> renderedImage;

	cv::Mat mK;
	int nFrames;
	bool state;

protected:

	Mapping * map;
	SysDesc * param;
	Viewer  * viewer;
	Tracker * tracker;
	std::thread * viewerThread;
	int num_frames_after_reloc;

};

#endif

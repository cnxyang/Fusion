#ifndef SYSTEM_HPP__
#define SYSTEM_HPP__

#include "Mapping.hpp"
#include "Viewer.hpp"
#include "Tracking.hpp"

class Viewer;

struct SysDesc {
	int cols, rows;
	float fx;
	float fy;
	float cx;
	float cy;
	float DepthCutoff;
	float DepthScale;
};

class System {
public:
	System(const char* str);
	System(SysDesc* pParam);
	void GrabImageRGBD(Mat& imRGB, Mat& imD);
	void SetParameters(SysDesc& desc);
	void RenderScene(Mat& img);

private:
	Mapping* mpMap;
	Viewer* mpViewer;
	Tracking* mpTracker;
	SysDesc* mpParam;
};

#endif

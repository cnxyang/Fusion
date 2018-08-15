#include "System.hpp"

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), mpParam(nullptr) {
	if(!str)
		return;
}

System::System(SysDesc* pParam):
mpMap(nullptr),
mpViewer(nullptr),
mpTracker(nullptr){

	mpMap = new Mapping();

	mpViewer = new Viewer();
	mpTracker = new Tracking();

	mpViewer->SetMap(mpMap);
	mpViewer->SetSystem(this);
	mpViewer->SetTracker(mpTracker);

	mpTracker->SetMap(mpMap);

	if(pParam) {
		mpParam = new SysDesc();
		memcpy((void*)mpParam, (void*)pParam, sizeof(SysDesc));
	}
	else {
		mpParam = new SysDesc();
		mpParam->DepthScale = 1000.0f;
		mpParam->DepthCutoff = 8.0f;
		mpParam->fx = 525.0f;
		mpParam->fy = 525.0f;
		mpParam->cx = 320.0f;
		mpParam->cy = 240.0f;
		mpParam->cols = 640;
		mpParam->rows = 480;
	}

	Frame::mDepthCutoff = mpParam->DepthCutoff;
	Frame::mDepthScale = mpParam->DepthScale;
}

void System::GrabImageRGBD(Mat& imRGB, Mat& imD) {

	mpTracker->Track(imRGB, imD);
}

void System::RenderScene(Mat& img) {

}

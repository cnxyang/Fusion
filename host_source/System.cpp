#include "System.hpp"
#include "Timer.hpp"

using namespace std;

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), mpParam(nullptr),
mptViewer(nullptr) {
	if(!str)
		System(static_cast<SysDesc*>(nullptr));
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
		mpParam->TrackModel = true;
	}

	mptViewer = new thread(&Viewer::Spin, mpViewer);

	Frame::mDepthScale = mpParam->DepthScale;
	Frame::mDepthCutoff = mpParam->DepthCutoff;
	Tracking::mbTrackModel = mpParam->TrackModel;
}

void System::GrabImageRGBD(Mat& imRGB, Mat& imD) {

	mpTracker->Track(imRGB, imD);
}

void System::PrintTimings() {

	Timer::PrintTiming();
}

#include "System.hpp"
#include "Timer.hpp"

using namespace cv;
using namespace std;

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), mpParam(nullptr),
mptViewer(nullptr), mbStop(false) {
	if(!str)
		System(static_cast<SysDesc*>(nullptr));
}

System::System(SysDesc* pParam):
mpMap(nullptr),
mpViewer(nullptr),
mpTracker(nullptr),
mbStop(false){

	mpMap = new Mapping();
	mpMap->AllocateDeviceMemory(MapDesc());

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

	mK = Mat::eye(3, 3, CV_32FC1);
	mK.at<float>(0, 0) = mpParam->fx;
	mK.at<float>(1, 1) = mpParam->fy;
	mK.at<float>(0, 2) = mpParam->cx;
	mK.at<float>(1, 2) = mpParam->cy;
	Frame::SetK(mK);

	mptViewer = new thread(&Viewer::Spin, mpViewer);

	Frame::mDepthScale = mpParam->DepthScale;
	Frame::mDepthCutoff = mpParam->DepthCutoff;
}

void System::GrabImageRGBD(Mat& imRGB, Mat& imD) {

	bool bOK = mpTracker->Track(imRGB, imD);

	if (bOK) {
		int no = mpMap->FuseFrame(mpTracker->mLastFrame);
		Rendering rd;
		rd.cols = 640;
		rd.rows = 480;
		rd.fx = mK.at<float>(0, 0);
		rd.fy = mK.at<float>(1, 1);
		rd.cx = mK.at<float>(0, 2);
		rd.cy = mK.at<float>(1, 2);
		rd.Rview = mpTracker->mLastFrame.Rot_gpu();
		rd.invRview = mpTracker->mLastFrame.RotInv_gpu();
		rd.maxD = 5.0f;
		rd.minD = 0.1f;
		rd.tview = mpTracker->mLastFrame.Trans_gpu();

		mpMap->RenderMap(rd, no);
		mpTracker->AddObservation(rd);
		Mat tmp(rd.rows, rd.cols, CV_8UC4);
		rd.Render.download((void*) tmp.data, tmp.step);
		resize(tmp, tmp, cv::Size(tmp.cols * 2, tmp.rows * 2));
		imshow("img", tmp);
	}

	if(mbStop)
		exit(0);
}

void System::Reboot() {
	mpMap->ResetDeviceMemory();
	mpTracker->ResetTracking();
}

void System::PrintTimings() {
	Timer::Print();
}

void System::Stop() {
	mbStop = true;
}

void System::JoinViewer() {

	while(!mbStop) {
		usleep(3000);
	}
}

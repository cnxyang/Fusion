#include "System.hpp"
#include "Timer.hpp"

using namespace cv;
using namespace std;

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), mpParam(nullptr),
mptViewer(nullptr), mbStop(false),
nFrames(0) {
	if(!str)
		System(static_cast<SysDesc*>(nullptr));
}

System::System(SysDesc* pParam):
mpMap(nullptr),
mpViewer(nullptr),
mpTracker(nullptr),
mbStop(false),
nFrames(0){

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

	mpMap = new Mapping();
	mpMap->AllocateDeviceMemory();

	mpViewer = new Viewer();
	mpTracker = new Tracking(mpParam->cols,
							 mpParam->rows,
							 mpParam->fx,
							 mpParam->fy,
							 mpParam->cx,
							 mpParam->cy);

	mpViewer->SetMap(mpMap);
	mpViewer->SetSystem(this);
	mpViewer->SetTracker(mpTracker);

	mpTracker->SetMap(mpMap);

	mptViewer = new thread(&Viewer::Spin, mpViewer);

	Frame::mDepthScale = mpParam->DepthScale;
	Frame::mDepthCutoff = mpParam->DepthCutoff;
//	Timer::Enable();
}

void System::GrabImageRGBD(Mat& imRGB, Mat& imD) {

	bool bOK = mpTracker->grabFrame(imRGB, imD);

	if (bOK) {
		uint no;
		mpMap->FuseDepthAndColor(mpTracker->lastDepth[0], mpTracker->color,
				mpTracker->mLastFrame.Rot_gpu(),
				mpTracker->mLastFrame.RotInv_gpu(),
				mpTracker->mLastFrame.Trans_gpu(),
				Frame::fx(0), Frame::fy(0),
				Frame::cx(0), Frame::cy(0),
				0.1f, 3.0f, no);

		mpMap->RayTrace(no, mpTracker->mLastFrame.Rot_gpu(),
				mpTracker->mLastFrame.RotInv_gpu(),
				mpTracker->mLastFrame.Trans_gpu(), mpTracker->lastVMap[0],
				mpTracker->lastNMap[0]);

		if(nFrames > 30) {
			nFrames = 0;
			mpMap->MeshScene();
		}
		nFrames++;
	}

	if(mbStop)
		exit(0);
}

void System::SaveMesh() {

	uint n = mpMap->MeshScene();
//	float3 * host_tri = mpMap->mHostMesh;
//	FILE *f = fopen("scene.stl", "wb+");
//	if (f != NULL) {
//		for (int i = 0; i < 80; i++)
//			fwrite(" ", sizeof(char), 1, f);
//		fwrite(&n, sizeof(int), 1, f);
//		float zero = 0.0f;
//		short attribute = 0;
//		for (uint i = 0; i < n; i++) {
//			fwrite(&zero, sizeof(float), 1, f);
//			fwrite(&zero, sizeof(float), 1, f);
//			fwrite(&zero, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 0].x, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 0].y, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 0].z, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 1].x, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 1].y, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 1].z, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 2].x, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 2].y, sizeof(float), 1, f);
//			fwrite(&host_tri[i * 3 + 2].z, sizeof(float), 1, f);
//			fwrite(&attribute, sizeof(short), 1, f);
//		}
//		fclose(f);
//	}
//	delete host_tri;
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

	}
}

#include "timer.h"
#include "system.h"

#include <fstream>

using namespace cv;
using namespace std;

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), mpParam(nullptr),
mptViewer(nullptr), requestStop(false),
nFrames(0), requestReboot(false) {
	if(!str)
		System(static_cast<SysDesc*>(nullptr));
}

System::System(SysDesc* pParam) :
		mpMap(nullptr), mpViewer(nullptr), mpTracker(nullptr),
		requestStop(false),	nFrames(0), requestSaveMesh(false),
		requestReboot(false) {

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
	mpMap->create();

	mpViewer = new Viewer();
	mpTracker = new tracker(mpParam->cols,
							 mpParam->rows,
							 mpParam->fx,
							 mpParam->fy,
							 mpParam->cx,
							 mpParam->cy);

	mpViewer->setMap(mpMap);
	mpViewer->setSystem(this);
	mpViewer->setTracker(mpTracker);

	mpTracker->setMap(mpMap);

	mptViewer = new thread(&Viewer::spin, mpViewer);

	Frame::mDepthScale = mpParam->DepthScale;
	Frame::mDepthCutoff = mpParam->DepthCutoff;
	Timer::Enable();
}

bool System::grabImage(Mat& imRGB, Mat& imD) {

	if(requestSaveMesh) {
		saveMesh();
		mutexReq.lock();
		requestSaveMesh = false;
		mutexReq.unlock();
	}

	if(requestReboot) {
		reboot();
		mutexReq.lock();
		requestReboot = false;
		mutexReq.unlock();
	}

	if(requestStop) {
		mpViewer->signalQuit();
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());
		return false;
	}

	Timer::Start("all", "all");
	bool bOK = mpTracker->grabFrame(imRGB, imD);

	if (bOK) {
		uint no;
		mpMap->fuseColor(mpTracker->lastDepth[0], mpTracker->color,
				mpTracker->lastFrame.Rot_gpu(),
				mpTracker->lastFrame.RotInv_gpu(),
				mpTracker->lastFrame.Trans_gpu(), no);

		mpMap->rayTrace(no, mpTracker->lastFrame.Rot_gpu(),
				mpTracker->lastFrame.RotInv_gpu(),
				mpTracker->lastFrame.Trans_gpu(), mpTracker->lastVMap[0],
				mpTracker->lastNMap[0]);

		if(nFrames > 15) {
			nFrames = 0;
			mpMap->createModel();
		}
		nFrames++;
	}

	Timer::Stop("all", "all");
	Timer::Print();
	return true;
}

void System::saveMesh() {

	mpMap->createModel();

	float3 * host_vertex = (float3*) malloc(sizeof(float3) * mpMap->noTrianglesHost * 3);
	float3 * host_normal = (float3*) malloc(sizeof(float3) * mpMap->noTrianglesHost * 3);
	uchar3 * host_color = (uchar3*) malloc(sizeof(uchar3) * mpMap->noTrianglesHost * 3);
	mpMap->modelVertex.download(host_vertex, mpMap->noTrianglesHost * 3);
	mpMap->modelNormal.download(host_normal, mpMap->noTrianglesHost * 3);
	mpMap->modelColor.download(host_color, mpMap->noTrianglesHost * 3);

	std::ofstream file;
	file.open("scene.ply");
		file << "ply\n";
		file << "format ascii 1.0\n";
		file << "element vertex " << mpMap->noTrianglesHost * 3 << "\n";
		file << "property float x\n";
		file << "property float y\n";
		file << "property float z\n";
		file << "property float nx\n";
		file << "property float ny\n";
		file << "property float nz\n";
		file << "property uchar red\n";
		file << "property uchar green\n";
		file << "property uchar blue\n";
		file << "element face " << mpMap->noTrianglesHost << "\n";
		file << "property list uchar uint vertex_indices\n";
		file << "end_header" << std::endl;

	for (uint i = 0; i <  mpMap->noTrianglesHost * 3; ++i) {
		file << host_vertex[i].x << " "
			 << host_vertex[i].y << " "
			 << host_vertex[i].z << " "
		     << host_normal[i].x << " "
			 << host_normal[i].y << " "
			 << host_normal[i].z << " "
		     << (int) host_color[i].x << " "
			 << (int) host_color[i].y << " "
			 << (int) host_color[i].z << std::endl;
	}

	uchar numFaces = 3;
	for (uint i = 0; i <  mpMap->noTrianglesHost; ++i) {
		file << (static_cast<int>(numFaces) & 0xFF) << " "
			 << (int) i * 3 + 0 << " "
			 << (int) i * 3 + 1 << " "
			 << (int) i * 3 + 2 << std::endl;
	}

	file.close();
	delete host_vertex;
	delete host_normal;
	delete host_color;
}

void System::reboot() {
	mpMap->reset();
	mpTracker->reset();
}

void System::joinViewer() {

	while(true) {

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		if(requestSaveMesh) {
			saveMesh();
			mutexReq.lock();
			requestSaveMesh = false;
			mutexReq.unlock();
		}

		if(requestStop) {
			return;
		}
	}
}

#include "timer.h"
#include "system.h"
#include "cufunc.h"
#include <fstream>

using namespace cv;
using namespace std;

System::System(const char* str):
mpMap(nullptr), mpViewer(nullptr),
mpTracker(nullptr), param(nullptr),
mptViewer(nullptr), requestStop(false),
nFrames(0), requestReboot(false) {
	if(!str)
		System(static_cast<SysDesc*>(nullptr));
}

System::System(SysDesc* pParam) :
		mpMap(nullptr), mpViewer(nullptr), mpTracker(nullptr),
		requestStop(false),	nFrames(0), requestSaveMesh(false),
		requestReboot(false), paused(false), state(true) {

	if(pParam) {
		param = new SysDesc();
		memcpy((void*)param, (void*)pParam, sizeof(SysDesc));
	}
	else {
		param = new SysDesc();
		param->DepthScale = 1000.0f;
		param->DepthCutoff = 8.0f;
		param->fx = 525.0f;
		param->fy = 525.0f;
		param->cx = 320.0f;
		param->cy = 240.0f;
		param->cols = 640;
		param->rows = 480;
		param->TrackModel = true;
	}

	mK = Mat::eye(3, 3, CV_32FC1);
	mK.at<float>(0, 0) = param->fx;
	mK.at<float>(1, 1) = param->fy;
	mK.at<float>(0, 2) = param->cx;
	mK.at<float>(1, 2) = param->cy;
	Frame::SetK(mK);

	mpMap = new Mapping();
	mpMap->create();

	mpViewer = new Viewer();
	mpTracker = new Tracker(param->cols,
							 param->rows,
							 param->fx,
							 param->fy,
							 param->cx,
							 param->cy);

	mpViewer->setMap(mpMap);
	mpViewer->setSystem(this);
	mpViewer->setTracker(mpTracker);

	mpTracker->setMap(mpMap);

//	mptViewer = new thread(&Viewer::spin, mpViewer);
//	mptViewer->detach();

	Frame::mDepthScale = param->DepthScale;
	Frame::mDepthCutoff = param->DepthCutoff;
	Timer::Enable();
}

bool System::grabImage(const Mat & image, const Mat & depth) {

	if(requestSaveMesh) {
		saveMesh();
		requestSaveMesh = false;
	}

	if(requestReboot) {
		reboot();
		requestReboot = false;
	}

	if(requestStop) {
		mpViewer->signalQuit();
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());
		return false;
	}

	Timer::Start("all", "all");
	if(!paused) {
		state = mpTracker->grabFrame(image, depth);
	}

	if (state) {
		uint no;
		Timer::Start("fuse", "fuse");
		mpMap->fuseColor(mpTracker->lastDepth[0], mpTracker->color,
				mpTracker->lastFrame.Rot_gpu(),
				mpTracker->lastFrame.RotInv_gpu(),
				mpTracker->lastFrame.Trans_gpu(), no);
		SafeCall(cudaDeviceSynchronize());
		Timer::Stop("fuse", "fuse");

//		if(nFrames % 5 != 0) {
//			Eigen::Matrix4d Tlastcurr = mpTracker->lastFrame.pose.inverse() * mpTracker->nextFrame.pose;
//			Eigen::Matrix3d Rlastcurr = Tlastcurr.inverse().topLeftCorner(3, 3);
//			Eigen::Vector3d tlastcurr = Tlastcurr.inverse().topRightCorner(3, 1);
//			std::cout << Tlastcurr << std::endl;
//			Matrix3f deviceR;
//			deviceR.rowx = { (float) Rlastcurr(0, 0), (float) Rlastcurr(0, 1), (float) Rlastcurr(0, 2) };
//			deviceR.rowy = { (float) Rlastcurr(1, 0), (float) Rlastcurr(1, 1), (float) Rlastcurr(1, 2) };
//			deviceR.rowz = { (float) Rlastcurr(2, 0), (float) Rlastcurr(2, 1), (float) Rlastcurr(2, 2) };
//			float3 devicet = { (float) tlastcurr(0), (float) tlastcurr(1), (float) tlastcurr(2) };
//			forwardProjection(mpTracker->nextVMap[0], mpTracker->nextNMap[0],
//					mpTracker->lastVMap[0], mpTracker->lastNMap[0],
//					mpTracker->lastFrame.Rot_gpu(),
//					mpTracker->lastFrame.Trans_gpu(),
//					mpTracker->nextFrame.RotInv_gpu(),
//					mpTracker->nextFrame.Trans_gpu(), Frame::fx(0),
//					Frame::fy(0), Frame::cx(0), Frame::cy(0));

//			mpTracker->lastVMap[0].swap(mpTracker->nextVMap[0]);
//			mpTracker->lastNMap[0].swap(mpTracker->nextNMap[0]);

//			cv::Mat img(480, 640, CV_32FC3);
//			mpTracker->nextNMap[0].download(img.data, img.step);
//			cv::imshow("img", img);
//			cv::waitKey(0);
//		}
//		else {
			Timer::Start("raytracing", "raytracing");
			mpMap->rayTrace(no, mpTracker->lastFrame.Rot_gpu(),
					mpTracker->lastFrame.RotInv_gpu(),
					mpTracker->lastFrame.Trans_gpu(), mpTracker->lastVMap[0],
					mpTracker->lastNMap[0]);
//		}
			Timer::Stop("raytracing", "raytracing");

		cv::Mat img(480, 640, CV_8UC4);
		mpTracker->renderedImage.download(img.data, img.step);
		cv::imshow("img", img);
		int key = cv::waitKey(10);
		if(key == 27)
			return false;

		if(nFrames % 25 == 0) {
			mpMap->createModel();
			mpMap->updateMapKeys();
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

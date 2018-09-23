#include "System.h"
#include <fstream>

using namespace cv;
using namespace std;

Matrix3f eigen_to_mat3f(Eigen::Matrix3d mat) {
	Matrix3f mat3f;
	mat3f.rowx = make_float3((float) mat(0, 0), (float) mat(0, 1), (float)mat(0, 2));
	mat3f.rowy = make_float3((float) mat(1, 0), (float) mat(1, 1), (float)mat(1, 2));
	mat3f.rowz = make_float3((float) mat(2, 0), (float) mat(2, 1), (float)mat(2, 2));
	return mat3f;
}

float3 eigen_to_float3(Eigen::Vector3d vec) {
	return make_float3((float) vec(0), (float) vec(1), (float) vec(2));
}

System::System(SysDesc* pParam) :
		map(0), viewer(0), tracker(0), requestStop(false), nFrames(0),
		requestSaveMesh(false), requestReboot(false), paused(false),
		state(true), requestMesh(false) {

	if(pParam) {
		param = new SysDesc();
		memcpy((void*) param, (void*) pParam, sizeof(SysDesc));
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

	map = new Mapping();
	map->Create();

	viewer = new Viewer();
	tracker = new Tracker(param->cols, param->rows,
			param->fx, param->fy, param->cx, param->cy);

	viewer->setMap(map);
	viewer->setSystem(this);
	viewer->setTracker(tracker);

	tracker->SetMap(map);

	viewerThread = new thread(&Viewer::spin, viewer);
	viewerThread->detach();

	Frame::mDepthScale = param->DepthScale;
	Frame::mDepthCutoff = param->DepthCutoff;

	vmap.create(param->cols, param->rows);
	nmap.create(param->cols, param->rows);
	renderedImage.create(param->cols, param->rows);
	num_frames_after_reloc = 10;
}

bool System::GrabImage(const Mat & image, const Mat & depth) {

	if(requestSaveMesh) {
		SaveMesh();
		requestSaveMesh = false;
	}

	if(requestReboot) {
		Reboot();
		requestReboot = false;
	}

	if(requestStop) {
		viewer->signalQuit();
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());
		return false;
	}

	state = tracker->GrabFrame(image, depth);

	switch(tracker->state) {
	case 0:
		num_frames_after_reloc++;
		break;

	case -1:
		num_frames_after_reloc = 0;
		break;
	}

	if (state) {
		uint noBlocks;
		if (!tracker->mappingDisabled && tracker->state != -1 && num_frames_after_reloc >= 10)
			map->FuseColor(tracker->LastFrame, noBlocks);

		if (!tracker->mappingDisabled && tracker->state != -1) {
			map->RayTrace(noBlocks, tracker->LastFrame);
		} else {
			map->UpdateVisibility(tracker->LastFrame, noBlocks);
			map->RayTrace(noBlocks, tracker->LastFrame);
		}

		if(nFrames % 25 == 0 && requestMesh) {
			if(!tracker->mappingDisabled) {
				map->CreateModel();
				map->UpdateMapKeys();
			}
		}

		//		Eigen::AngleAxisd angle(M_PI / 2, -Eigen::Vector3d::UnitX());
		//		Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
		//		Matrix3f curot = eigen_to_mat3f(angle.toRotationMatrix());
		//		Matrix3f curotinv = eigen_to_mat3f(angle.toRotationMatrix().transpose());
		//		float3 trans = make_float3(0, -8, 0);
		//		mpMap->updateVisibility(curot, curotinv, trans, 8.0, 11.0, Frame::fx(0),
		//				Frame::fy(0), vmap.cols / 2, vmap.rows / 2, no);
		//		mpMap->rayTrace(no, curot, curotinv, trans, vmap, nmap, 8.0, 11.0,
		//				Frame::fx(0), Frame::fy(0), vmap.cols / 2, vmap.rows / 2);
		//		RenderImage(vmap, nmap, make_float3(0, 0, 0), renderedImage);

		//		cv::Mat img(480, 640, CV_8UC4);
		//		image.download(img.data, img.step);
		//		cv::imshow("img", img);
		//		int key = cv::waitKey(10);
		//		if(key == 27)
		//			return false;
		//
		//		imageUpdated = true;

		nFrames++;
	}

	return true;
}

void System::SaveMesh() {

	map->CreateModel();

	float3 * host_vertex = (float3*) malloc(sizeof(float3) * map->noTrianglesHost * 3);
	float3 * host_normal = (float3*) malloc(sizeof(float3) * map->noTrianglesHost * 3);
	uchar3 * host_color = (uchar3*) malloc(sizeof(uchar3) * map->noTrianglesHost * 3);
	map->modelVertex.download(host_vertex, map->noTrianglesHost * 3);
	map->modelNormal.download(host_normal, map->noTrianglesHost * 3);
	map->modelColor.download(host_color, map->noTrianglesHost * 3);

	std::ofstream file;
	file.open("/home/xyang/scene.ply");
		file << "ply\n";
		file << "format ascii 1.0\n";
		file << "element vertex " << map->noTrianglesHost * 3 << "\n";
		file << "property float x\n";
		file << "property float y\n";
		file << "property float z\n";
		file << "property float nx\n";
		file << "property float ny\n";
		file << "property float nz\n";
		file << "property uchar red\n";
		file << "property uchar green\n";
		file << "property uchar blue\n";
		file << "element face " << map->noTrianglesHost << "\n";
		file << "property list uchar uint vertex_indices\n";
		file << "end_header" << std::endl;

	for (uint i = 0; i <  map->noTrianglesHost * 3; ++i) {
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
	for (uint i = 0; i <  map->noTrianglesHost; ++i) {
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

void System::Reboot() {
	map->Reset();
	tracker->ResetTracking();
}

void System::JoinViewer() {

	while(true) {

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		if(requestSaveMesh) {
			SaveMesh();
			requestSaveMesh = false;
		}

		if(requestStop) {
			return;
		}
	}
}

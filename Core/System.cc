#include "System.h"
#include <fstream>

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
		state(true), requestMesh(false), requestSaveMap(false),
		requestReadMap(false) {

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

	mK = cv::Mat::eye(3, 3, CV_32FC1);
	mK.at<float>(0, 0) = param->fx;
	mK.at<float>(1, 1) = param->fy;
	mK.at<float>(0, 2) = param->cx;
	mK.at<float>(1, 2) = param->cy;
	Frame::SetK(mK);

	map = new Mapping();
	map->Create();

	optimizer = new Optimizer();
	viewer = new Viewer();
	tracker = new Tracker(param->cols, param->rows,
			param->fx, param->fy, param->cx, param->cy);

	viewer->setMap(map);
	viewer->setSystem(this);
	viewer->setTracker(tracker);

	tracker->SetMap(map);
	viewerThread = new std::thread(&Viewer::spin, viewer);
	viewerThread->detach();

	optimizer->SetMap(map);
	optimizerThd = new std::thread(&Optimizer::run, optimizer);
	optimizerThd->detach();

	Frame::mDepthScale = param->DepthScale;
	Frame::mDepthCutoff = param->DepthCutoff;

	vmap.create(param->cols, param->rows);
	nmap.create(param->cols, param->rows);
	renderedImage.create(param->cols, param->rows);
	num_frames_after_reloc = 10;
}

void System::RenderTopDown(float dist) {

	uint noBlocks;
	Eigen::AngleAxisd angle(M_PI / 2, -Eigen::Vector3d::UnitX());
	Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
	Matrix3f curot = eigen_to_mat3f(angle.toRotationMatrix());
	Matrix3f curotinv = eigen_to_mat3f(angle.toRotationMatrix().transpose());
	float3 trans = make_float3(0, -dist, 0);
	map->UpdateVisibility(curot, curotinv, trans, dist + DeviceMap::DepthMin,
			dist + DeviceMap::DepthMax, Frame::fx(0), Frame::fy(0),
			vmap.cols / 2, vmap.rows / 2, noBlocks);
	map->RayTrace(noBlocks, curot, curotinv, trans, vmap, nmap,
			dist + DeviceMap::DepthMin, dist + DeviceMap::DepthMax,
			Frame::fx(0), Frame::fy(0), vmap.cols / 2, vmap.rows / 2);
	RenderImage(vmap, nmap, make_float3(0), renderedImage);
}

bool System::GrabImage(const cv::Mat & image, const cv::Mat & depth) {

	FilterMessage();

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
		if (!tracker->mappingDisabled && tracker->state != -1
				&& num_frames_after_reloc >= 10)
			map->FuseColor(tracker->LastFrame, noBlocks);

		if (!tracker->mappingDisabled && tracker->state != -1) {
			map->RayTrace(noBlocks, tracker->LastFrame);
		} else {
			map->UpdateVisibility(tracker->LastFrame, noBlocks);
			map->RayTrace(noBlocks, tracker->LastFrame);
		}

		if (nFrames % 25 == 0 && requestMesh) {
			if (!tracker->mappingDisabled) {
				map->CreateModel();
				map->UpdateMapKeys();
			}
		}

		if(!requestMesh) {
			RenderTopDown(8.0f);
		}

		imageUpdated = true;

		nFrames++;
	}

	return true;
}

void System::WriteMeshToDisk() {

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

void System::WriteMapToDisk() {

	map->DownloadToRAM();

	auto file = std::fstream("/home/xyang/map.bin", std::ios::out | std::ios::binary);

	const int NumSdfBlocks = DeviceMap::NumSdfBlocks;
	const int NumBuckets = DeviceMap::NumBuckets;
	const int NumVoxels = DeviceMap::NumVoxels;
	const int NumEntries = DeviceMap::NumEntries;

	// begin writing of general map info
	file.write((const char*)&NumSdfBlocks, sizeof(int));
	file.write((const char*)&NumBuckets, sizeof(int));
	file.write((const char*)&NumVoxels, sizeof(int));
	file.write((const char*)&NumEntries, sizeof(int));

	// begin writing of dense map
	file.write((char*) map->heapCounterRAM, sizeof(int));
	file.write((char*) map->hashCounterRAM, sizeof(int));
	file.write((char*) map->noVisibleEntriesRAM, sizeof(uint));
	file.write((char*) map->heapRAM, sizeof(int) * DeviceMap::NumSdfBlocks);
	file.write((char*) map->bucketMutexRAM, sizeof(int) * DeviceMap::NumBuckets);
	file.write((char*) map->sdfBlockRAM, sizeof(Voxel) * DeviceMap::NumVoxels);
	file.write((char*) map->hashEntriesRAM, sizeof(HashEntry) * DeviceMap::NumEntries);
	file.write((char*) map->visibleEntriesRAM, sizeof(HashEntry) * DeviceMap::NumEntries);

	// begin writing of feature map
	file.write((char*) map->mutexKeysRAM, sizeof(int) * KeyMap::MaxKeys);
	file.write((char*) map->mapKeysRAM, sizeof(SURF) * KeyMap::maxEntries);

	// clean up
	file.close();
	map->ReleaseRAM();
}

void System::ReadMapFromDisk() {

	int NumSdfBlocks;
	int NumBuckets;
	int NumVoxels;
	int NumEntries;

	auto file = std::fstream("/home/xyang/map.bin", std::ios::in | std::ios::binary);

	// begin reading of general map info
	file.read((char *) &NumSdfBlocks, sizeof(int));
	file.read((char *) &NumBuckets, sizeof(int));
	file.read((char *) &NumVoxels, sizeof(int));
	file.read((char *) &NumEntries, sizeof(int));

	map->CreateRAM();

	// begin reading of dense map
	file.read((char*) map->heapCounterRAM, sizeof(int));
	file.read((char*) map->hashCounterRAM, sizeof(int));
	file.read((char*) map->noVisibleEntriesRAM, sizeof(uint));
	file.read((char*) map->heapRAM, sizeof(int) * DeviceMap::NumSdfBlocks);
	file.read((char*) map->bucketMutexRAM, sizeof(int) * DeviceMap::NumBuckets);
	file.read((char*) map->sdfBlockRAM, sizeof(Voxel) * DeviceMap::NumVoxels);
	file.read((char*) map->hashEntriesRAM, sizeof(HashEntry) * DeviceMap::NumEntries);
	file.read((char*) map->visibleEntriesRAM, sizeof(HashEntry) * DeviceMap::NumEntries);

	// begin reading of feature map
	file.read((char*) map->mutexKeysRAM, sizeof(int) * KeyMap::MaxKeys);
	file.read((char*) map->mapKeysRAM, sizeof(SURF) * KeyMap::maxEntries);

	map->UploadFromRAM();
	map->ReleaseRAM();

	file.close();

	map->CreateModel();
	tracker->mappingDisabled = true;
	tracker->state = 1;
	tracker->lastState = 1;
}

void System::RebootSystem() {

	map->Reset();

	tracker->ResetTracking();
}

void System::FilterMessage(bool finished) {

	if(requestSaveMesh) {
		WriteMeshToDisk();
		requestSaveMesh = false;
	}

	if(!finished && requestReboot) {
		RebootSystem();
		requestReboot = false;
	}

	if(requestSaveMap) {
		WriteMapToDisk();
		requestSaveMap = false;
	}

	if(requestReadMap) {
		ReadMapFromDisk();
		tracker->ResetTracking();
		requestReadMap = false;
	}

	if(requestStop) {
		viewer->signalQuit();
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());
		exit(0);
	}
}

void System::JoinViewer() {

	while(true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		FilterMessage(true);
	}
}

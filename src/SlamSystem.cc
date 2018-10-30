#include "SlamSystem.h"
#include "Legacy/Tracking.h"
#include "GlWrapper/GlViewer.h"
#include "ICPTracking/PointCloud.h"
#include "ICPTracking/ICPTracker.h"
#include "GlobalMapping/DenseMap.h"
#include "Utilities/EigenUtils.h"
#include <fstream>
#include <unordered_set>

void SlamSystem::renderBirdsEyeView(float dist) {

	uint noBlocks;
	Eigen::AngleAxisd angle(M_PI / 2, -Eigen::Vector3d::UnitX());
	Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
	Matrix3f curot = EigenToMatrix3f(angle.toRotationMatrix());
	Matrix3f curotinv = EigenToMatrix3f(angle.toRotationMatrix().transpose());
	float3 trans = make_float3(0, -dist, 0);
	map->UpdateVisibility(curot, curotinv, trans, dist + DeviceMap::DepthMin,
			dist + DeviceMap::DepthMax, Frame::fx(0), Frame::fy(0),
			vmap.cols / 2, vmap.rows / 2, noBlocks);
	map->RayTrace(noBlocks, curot, curotinv, trans, vmap, nmap,
			dist + DeviceMap::DepthMin, dist + DeviceMap::DepthMax,
			Frame::fx(0), Frame::fy(0), vmap.cols / 2, vmap.rows / 2);
	RenderImage(vmap, nmap, make_float3(0), renderedImage);
}



void SlamSystem::WriteMeshToDisk() {

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

void SlamSystem::WriteMapToDisk() {

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

void SlamSystem::ReadMapFromDisk() {

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

void SlamSystem::RebootSystem() {

	map->Reset();

	tracker->ResetTracking();
}

void SlamSystem::processMessages(bool finished) {

	if(requestSaveMesh)
	{
		WriteMeshToDisk();
		requestSaveMesh = false;
	}

	if(!finished && requestReboot)
	{
		RebootSystem();
		requestReboot = false;
	}

	if(requestSaveMap)
	{
		WriteMapToDisk();
		requestSaveMap = false;
	}

	if(requestReadMap)
	{
		ReadMapFromDisk();
		tracker->ResetTracking();
		requestReadMap = false;
	}
}

// =============== REFACTORING ====================

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K) :
	width(w), height(h), K(K), dumpMapToDisk(false),
	numImagesProcessed(0), keepRunning(true), trackingTarget(NULL),
	trackingReference(NULL), currentKeyFrame(NULL),
	systemRunning(true), latestTrackedFrame(NULL)
{
	Frame::SetK(K);

	map = new DenseMap();
	map->allocateDeviceMemory();

	viewer = new GlViewer("SlamSystem", w, h, K);
	tracker = new Tracker(w, h, K(0, 0), K(1, 1), K(0, 2), K(1, 2));

	viewer->setMap(map);
	viewer->setSystem(this);
	tracker->SetMap(map);

	Frame::mDepthScale = 1000.f;
	Frame::mDepthCutoff = 3.0f;

	vmap.create(w, h);
	nmap.create(w, h);
	renderedImage.create(w, h);

	trackingReference = new PointCloud();
	trackingTarget = new PointCloud();
	keyFrameGraph = new KeyFrameGraph(w, h, K);
	threadConstraintSearch = std::thread(&SlamSystem::loopConstraintSearch, this);
	threadVisualisation = std::thread(&SlamSystem::loopVisualisation, this);

	mainTracker = new ICPTracker(w, h, K);
	mainTracker->setIterations({ 10, 5, 3 });
}

SlamSystem::~SlamSystem()
{
	keepRunning = false;

	printf("Waiting for all other threads to finish.\n");

	// waiting for all other threads
	threadConstraintSearch.join();

	printf("DONE Waiting for other threads.\n");

	// delete all resources
	delete tracker;
	delete map;
	delete trackingReference;
	delete trackingTarget;
	delete keyFrameGraph;

	printf("DONE Cleaning up.\n");
}

void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp) {

	// process messages before tracking
	// this include system reboot requests
	// and other system directives.
	processMessages();

	// create new frame and generate point cloud
	Frame* currentFrame = new Frame(img, depth, id, K, timeStamp);
	trackingTarget->generateCloud(currentFrame);
	viewer->setCurrentImages(trackingTarget);

	// first frame of the dataset
	// no reference frame to track to
	if(!trackingReference->frame)
	{
		// for efficiency reasons swap tracking data
		// only used when doing frame-to-frame tracking
		std::swap(trackingReference, trackingTarget);

		// do a raycast to ensure we have a copy of the map
		map->takeSnapShot(trackingReference);
		trackingReference->generatePyramid();
		latestTrackedFrame = currentFrame;
		return;
	}

	SE3 poseUpdate = mainTracker->trackSE3(trackingReference, trackingTarget);
	currentFrame->pose = latestTrackedFrame->pose * poseUpdate.inverse();
	latestTrackedFrame = currentFrame;

	std::cout << mainTracker->icpInlierRatio << std::endl;

	std::swap(trackingReference, trackingTarget);

	int numVisibleBlocks = map->fusePointCloud(trackingReference);
	map->takeSnapShot(trackingReference, numVisibleBlocks);


	if (numImagesProcessed % 25 == 0 && requestMesh)
	{
		map->CreateModel();
	}

	numImagesProcessed++;

	viewer->setCurrentCamPose(latestTrackedFrame->pose);
}

void SlamSystem::loopVisualisation()
{
	printf("Visualisation Thread Started.\n");
	viewer->enableGlContext();

	while(keepRunning)
	{
		while(!viewer->shouldQuit())
		{
			viewer->processMessages();
			viewer->drawViewsToScreen();
		}

		systemRunning = false;
	}

	printf("Visualisation Thread Exited.\n");
}

void SlamSystem::loopOptimization()
{
	printf("Optimisation Thread Started.\n");

	while(keepRunning)
	{

	}

	printf("Optimisation Thread Started.\n");
}

void SlamSystem::loopConstraintSearch()
{
	printf("Constraint Search Thread Started.\n");

	// Used for constraint searching
	PointCloud* firstFrame = new PointCloud();
	PointCloud* secondFrame = new PointCloud();

	std::unique_lock<std::mutex> lock(newKeyFrameMutex);

	while(keepRunning)
	{
		if(newKeyFrames.size() == 0)
		{
			lock.unlock();

			lock.lock();
		}
		else
		{
			Frame* newKF = newKeyFrames.front();
			newKeyFrames.pop_front();
			lock.unlock();
			findConstraintsForNewKeyFrames(newKF);
			lock.lock();
		}
	}

	printf("Constraint Search Thread Exited.\n");
}

void SlamSystem::findConstraintsForNewKeyFrames(Frame* newKF)
{
	std::vector<KFConstraint*> constraints;
	std::unordered_set<Frame*, std::hash<Frame*>> candidates;
	candidates = keyFrameGraph->searchCandidates(newKF);

	// Erase ones that are already neighbours
	for(std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	{
		if(newKF->neighbors.find(*c) != newKF->neighbors.end())
			c = candidates.erase(c);
		else
			++c;
	}
}

void SlamSystem::updateVisualisation()
{

}

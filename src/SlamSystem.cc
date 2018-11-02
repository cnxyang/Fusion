#include "SlamSystem.h"
#include "Frame.h"
#include "GlViewer.h"
#include "PointCloud.h"
#include "ICPTracker.h"
#include "VoxelMap.h"
#include "EigenUtils.h"
#include "Settings.h"

#include <unordered_set>
#include <fstream>
#include <chrono>

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K) :
	width(w), height(h), K(K), keepRunning(true),
	trackingTarget(NULL), trackingReference(NULL),
	currentKeyFrame(NULL), systemRunning(true),
	latestTrackedFrame(NULL)
{
	map = new VoxelMap();
	map->allocateDeviceMemory();

	tracker = new ICPTracker(w, h, K);
	tracker->setIterations({ 10, 5, 3 });

	constraintTracker = new ICPTracker(w, h, K);
	constraintTracker->setIterations({ 5, 3, 1 });

	viewer = new GlViewer("SlamSystem", w, h, K);
	viewer->setSlamSystem(this);
	viewer->setVoxelMap(map);

	trackingReference = new PointCloud();
	trackingTarget = new PointCloud();
	keyFrameGraph = new KeyFrameGraph(w, h, K);

	threadConstraintSearch = std::thread(&SlamSystem::loopConstraintSearch, this);
	threadVisualisation = std::thread(&SlamSystem::loopVisualisation, this);

	CONSOLE("SLAM System Successfully Initiated.");
}

SlamSystem::~SlamSystem()
{
	CONSOLE("Waiting for All Other Threads to Finish.");
	keepRunning = false;

	// waiting for all other threads
	threadConstraintSearch.join();

	CONSOLE("DONE Waiting for Other Threads.");

	// delete all resources
	delete map;
	delete tracker;
	delete trackingReference;
	delete trackingTarget;
	delete keyFrameGraph;

	CONSOLE("DONE Cleaning Up.");
}

//void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp) {
//
//	// process messages before tracking
//	// this include system reboot requests
//	// and other system directives.
//	processMessages();
//
//	// create new frame and generate point cloud
//	Frame* currentFrame = new Frame(img, depth, id, K, timeStamp);
//	trackingTarget->generateCloud(currentFrame);
//	viewer->setCurrentImages(trackingTarget);
//
//	// first frame of the dataset
//	// no reference frame to track to
//	if(!trackingReference->frame)
//	{
//		// for efficiency reasons swap tracking data
//		// only used when doing frame-to-frame tracking
//		std::swap(trackingReference, trackingTarget);
//
//		// do a raycast to ensure we have a copy of the map
//		int numVisibleBlocks = map->fusePointCloud(trackingReference);
//		map->takeSnapShot(trackingReference, numVisibleBlocks);
//		trackingReference->generatePyramid();
//		latestTrackedFrame = currentFrame;
//		currentKeyFrame = currentFrame;
//		viewer->keyFrameGraph.push_back(currentKeyFrame->pose());
//		return;
//	}
//
//	// track current frame. return frame-to-last-frame transform
//	SE3 poseUpdate = tracker->trackSE3(trackingReference, trackingTarget, SE3(), false);
//
//	// Update current frame pose
//	currentFrame->pose() = latestTrackedFrame->pose() * poseUpdate.inverse();
//
//	// set last tracked frame to current frame
//	latestTrackedFrame = currentFrame;
//
//	// do a ray trace to ensure we have a copy of the map
//	int numVisibleBlocks = map->fusePointCloud(trackingTarget);
//	map->takeSnapShot(trackingTarget, numVisibleBlocks);
//	trackingTarget->generatePyramid();
//
//	// Update visualisation
//	updateVisualisation();
//
//	// for efficiency reasons swap tracking data
//	// only used when doing frame-to-frame tracking
//	std::swap(trackingReference, trackingTarget);
//
//	++systemState.numTrackedFrames;
//}

void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp)
{
	// process messages before tracking
	// this include system reboot requests
	// and other system directives.
	processMessages();

	// create new frame and generate point cloud
	Frame* currentFrame = new Frame(img, depth, id, K, timeStamp);

	trackingTarget->generateCloud(currentFrame);

	// first frame of the dataset
	// no reference frame to track to
	if(!trackingReference->frame)
	{
		// for efficiency reasons swap tracking data
		// only used when doing frame-to-frame tracking
		std::swap(trackingReference, trackingTarget);

		// do a raycast to ensure we have a copy of the map
		int numVisibleBlocks = map->fusePointCloud(trackingReference);
		map->takeSnapShot(trackingReference, numVisibleBlocks);
		trackingReference->generatePyramid();
		latestTrackedFrame = currentFrame;
		currentKeyFrame = currentFrame;
		viewer->keyFrameGraph.push_back(currentKeyFrame->pose());
		return;
	}

	// Initial pose estimate
	SE3 initialEstimate = latestTrackedFrame->pose().inverse() * currentKeyFrame->pose();

	// Track current frame
	// Return frame to current *Key Frame* transform
	SE3 poseUpdate = tracker->trackSE3(trackingReference, trackingTarget, initialEstimate, false);

	// Update current frame pose
	currentFrame->pose() = currentKeyFrame->pose() * poseUpdate.inverse();

	// Set last tracked frame to current frame
	currentFrame->poseStruct->parentPose = latestTrackedFrame->poseStruct;
	latestTrackedFrame = currentFrame;

	// Do a raycast to ensure we have a copy of the map
	int numVisibleBlocks = map->fusePointCloud(trackingTarget);

	// Update visualisation
	updateVisualisation();

	// check if we insert a new key frame
	if(tracker->trackingWasGood)
	{
		float pointUsage = std::min(tracker->icpInlierRatio, tracker->rgbInlierRatio);
		Sophus::Vector3d dist = poseUpdate.translation();

		float closenessScore = 16 * dist.norm() + (1 - pointUsage) * (1 - pointUsage);

		if(closenessScore > 2.f)
		{
			std::swap(trackingReference, trackingTarget);
			map->takeSnapShot(trackingReference);
			trackingReference->generatePyramid();
			currentKeyFrame = currentFrame;
			viewer->keyFrameGraph.push_back(currentKeyFrame->pose());
			++systemState.numTrackedKeyFrames;
		}
	}

	++systemState.numTrackedFrames;
}

void SlamSystem::processMessages()
{
	std::unique_lock<std::mutex> lock(messageQueueMutex);
	Msg newmsg = messageQueue.size() == 0 ? Msg(Msg::EMPTY_MSG) : messageQueue.front();

	switch(newmsg.data)
	{
	case Msg::SYSTEM_RESET:
		rebootSystem();
		break;

	case Msg::SYSTEM_SHUTDOWN:
		systemRunning = false;
		break;

	case Msg::WRITE_BINARY_MAP_TO_DISK:
		writeBinaryMapToDisk();
		break;

	case Msg::READ_BINARY_MAP_FROM_DISK:
		readBinaryMapFromDisk();
		break;

	case Msg::TOGGLE_MESH_ON:
		systemState.showGeneratedMesh = true;
		break;

	case Msg::TOGGLE_MESH_OFF:
		systemState.showGeneratedMesh = false;
		break;

	case Msg::TOGGLE_IMAGE_ON:
		systemState.showInputImages = true;
		break;

	case Msg::TOGGLE_IMAGE_OFF:
		systemState.showInputImages = false;
		break;

	default:
		return;
	}

	messageQueue.pop();
}

void SlamSystem::queueMessage(Msg newmsg)
{
	std::unique_lock<std::mutex> lock(messageQueueMutex);
	messageQueue.push(newmsg);
}

void SlamSystem::loopVisualisation()
{
	CONSOLE("Visualisation Thread Started.");
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

	CONSOLE("Visualisation Thread Exited.");
}

void SlamSystem::loopOptimization()
{
	CONSOLE("Optimisation Thread Started.");

	while(keepRunning)
	{

	}

	CONSOLE("Optimisation Thread Started.");
}

void SlamSystem::loopConstraintSearch()
{
	CONSOLE("Constraint Search Thread Started.");

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

	CONSOLE("Constraint Search Thread Exited.");
}

void SlamSystem::findConstraintsForNewKeyFrames(Frame* newKF)
{
	std::vector<KFConstraint*> constraints;
	std::unordered_set<Frame*, std::hash<Frame*>> candidates;
	candidates = keyFrameGraph->searchCandidates(newKF);

	// Erase ones that are already neighbours
	for (std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	{
		if(newKF->neighbors.find(*c) != newKF->neighbors.end())
			c = candidates.erase(c);
		else
			++c;
	}

	for (Frame* c : candidates)
	{

	}
}

void SlamSystem::tryTrackConstraint()
{

}

void SlamSystem::checkConstraints()
{

}

void SlamSystem::updateVisualisation()
{
	viewer->setCurrentCamPose(latestTrackedFrame->pose());

	if (systemState.showGeneratedMesh)
	{
		if (systemState.numTrackedFrames % 30 == 0)
		{
			map->CreateModel();
		}
	}

	if (systemState.showInputImages)
	{
		viewer->setCurrentImages(trackingTarget);
	}
}

void SlamSystem::exportMeshAsFile() {

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

void SlamSystem::writeBinaryMapToDisk() {

//	map->DownloadToRAM();
//
//	auto file = std::fstream("/home/xyang/map.bin", std::ios::out | std::ios::binary);
//
//	const int NumSdfBlocks = MapStruct::NumSdfBlocks;
//	const int NumBuckets = MapStruct::NumBuckets;
//	const int NumVoxels = MapStruct::NumVoxels;
//	const int NumEntries = MapStruct::NumEntries;
//
//	// begin writing of general map info
//	file.write((const char*)&NumSdfBlocks, sizeof(int));
//	file.write((const char*)&NumBuckets, sizeof(int));
//	file.write((const char*)&NumVoxels, sizeof(int));
//	file.write((const char*)&NumEntries, sizeof(int));
//
//	// begin writing of dense map
//	file.write((char*) map->heapCounterRAM, sizeof(int));
//	file.write((char*) map->hashCounterRAM, sizeof(int));
//	file.write((char*) map->noVisibleEntriesRAM, sizeof(uint));
//	file.write((char*) map->heapRAM, sizeof(int) * MapStruct::NumSdfBlocks);
//	file.write((char*) map->bucketMutexRAM, sizeof(int) * MapStruct::NumBuckets);
//	file.write((char*) map->sdfBlockRAM, sizeof(Voxel) * MapStruct::NumVoxels);
//	file.write((char*) map->hashEntriesRAM, sizeof(HashEntry) * MapStruct::NumEntries);
//	file.write((char*) map->visibleEntriesRAM, sizeof(HashEntry) * MapStruct::NumEntries);
//
//	// begin writing of feature map
//	file.write((char*) map->mutexKeysRAM, sizeof(int) * KeyMap::MaxKeys);
//	file.write((char*) map->mapKeysRAM, sizeof(SURF) * KeyMap::maxEntries);
//
//	// clean up
//	file.close();
//	map->ReleaseRAM();
}

void SlamSystem::readBinaryMapFromDisk() {

//	int NumSdfBlocks;
//	int NumBuckets;
//	int NumVoxels;
//	int NumEntries;
//
//	auto file = std::fstream("/home/xyang/map.bin", std::ios::in | std::ios::binary);
//
//	// begin reading of general map info
//	file.read((char *) &NumSdfBlocks, sizeof(int));
//	file.read((char *) &NumBuckets, sizeof(int));
//	file.read((char *) &NumVoxels, sizeof(int));
//	file.read((char *) &NumEntries, sizeof(int));
//
//	map->CreateRAM();
//
//	// begin reading of dense map
//	file.read((char*) map->heapCounterRAM, sizeof(int));
//	file.read((char*) map->hashCounterRAM, sizeof(int));
//	file.read((char*) map->noVisibleEntriesRAM, sizeof(uint));
//	file.read((char*) map->heapRAM, sizeof(int) * MapStruct::NumSdfBlocks);
//	file.read((char*) map->bucketMutexRAM, sizeof(int) * MapStruct::NumBuckets);
//	file.read((char*) map->sdfBlockRAM, sizeof(Voxel) * MapStruct::NumVoxels);
//	file.read((char*) map->hashEntriesRAM, sizeof(HashEntry) * MapStruct::NumEntries);
//	file.read((char*) map->visibleEntriesRAM, sizeof(HashEntry) * MapStruct::NumEntries);
//
//	// begin reading of feature map
//	file.read((char*) map->mutexKeysRAM, sizeof(int) * KeyMap::MaxKeys);
//	file.read((char*) map->mapKeysRAM, sizeof(SURF) * KeyMap::maxEntries);
//
//	map->UploadFromRAM();
//	map->ReleaseRAM();
//
//	file.close();
//
//	map->CreateModel();
}

void SlamSystem::rebootSystem()
{
	map->resetMapStruct();
	trackingReference->frame = NULL;
	viewer->keyFrameGraph.clear();
}

void SlamSystem::displayDebugImages(int ms)
{
	if(imageReference.empty())
	{
		imageReference.create(480, 640, CV_8UC1);
		depthReference.create(480, 640, CV_32FC1);
		imageTarget.create(480, 640, CV_8UC1);
		depthTarget.create(480, 640, CV_32FC1);
		nmapReference.create(480, 640, CV_32FC4);
		nmapTarget.create(480, 640, CV_32FC4);
	}

	trackingReference->image[0].download(imageReference.data, imageReference.step);
	trackingReference->depth[0].download(depthReference.data, depthReference.step);
	trackingReference->nmap[0].download(nmapReference.data, nmapReference.step);
	trackingTarget->image[0].download(imageTarget.data, imageTarget.step);
	trackingTarget->depth[0].download(depthTarget.data, depthTarget.step);
	trackingTarget->nmap[0].download(nmapTarget.data, nmapTarget.step);

	cv::imshow("imageReference", imageReference);
	cv::imshow("depthReference", depthReference);
	cv::imshow("nmapReference", nmapReference);
	cv::imshow("imageTarget", imageTarget);
	cv::imshow("depthTarget", depthTarget);
	cv::imshow("nmapTarget", nmapTarget);

	cv::waitKey(ms);
}

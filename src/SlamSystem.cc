#include "Frame.h"
#include "GlViewer.h"
#include "VoxelMap.h"
#include "Settings.h"
#include "SlamSystem.h"
#include "PointCloud.h"
#include "ICPTracker.h"
#include "EigenUtils.h"
#include "KeyFrameGraph.h"

#include <chrono>
#include <fstream>
#include <unordered_set>

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K) :
	width(w), height(h), K(K), keepRunning(true), latestTrackedFrame(0),
	newConstraintAdded(false), trackingTarget(0), trackingReference(0),
	currentKeyFrame(0), systemRunning(true), doFinalOptimization(false),
	havePoseUpdate(false), currentFrame(0)
{
	map = new VoxelMap();
	map->allocateDeviceMap();

	tracker = new ICPTracker(w, h, K);
	tracker->setIterations({ 10, 5, 3 });

	constraintTracker = new ICPTracker(w, h, K);
	constraintTracker->setIterations({ 10, 5, 3 });

	viewer = new GlViewer("SlamSystem", w, h, K);
	viewer->setSlamSystem(this);
	viewer->setVoxelMap(map);

	trackingReference = new PointCloud();
	trackingTarget = new PointCloud();
	keyFrameGraph = new KeyFrameGraph(w, h, K);

	firstFrame = new PointCloud();
	secondFrame = new PointCloud();

	threadConstraintSearch = std::thread(&SlamSystem::loopConstraintSearch, this);
	threadVisualisation = std::thread(&SlamSystem::loopVisualisation, this);
	threadMapGeneration = std::thread(&SlamSystem::loopMapGeneration, this);
	threadOptimization = std::thread(&SlamSystem::loopOptimization, this);

	CONSOLE("SLAM System Successfully Initiated.");
}

SlamSystem::~SlamSystem()
{
	CONSOLE("Waiting for All Other Threads to Finish.");
	keepRunning = false;

	// waiting for all other threads
	threadConstraintSearch.join();
	threadMapGeneration.join();
	threadOptimization.join();

	CONSOLE("DONE Waiting for Other Threads.");

	// delete all resources
	delete map;
	delete tracker;
	delete trackingReference;
	delete trackingTarget;
	delete keyFrameGraph;

	CONSOLE("DONE Cleaning Up.");
}

// MAIN tracking thread
// This tracking method uses frame-to-map tracking
// and key frames are computed after new frames
// have been registered.
void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp)
{
	// Process MESSAGES BEFORE tracking
	// includes system reboot requests
	// and other system directives.
	processMessages();
//	updateMap(10);

	// Create NEW frame
	currentFrame = new Frame(img, depth, id, K, timeStamp);

	// Generate point cloud for the new frame
	trackingTarget->generateCloud(currentFrame);

	if(!trackingReference->frame)
	{
		// for efficiency reasons swap tracking data
		// only used when doing frame-to-frame tracking
		std::swap(trackingReference, trackingTarget);
		// Integrate current frame into the map
		int blockCount = map->fuseImages(trackingReference);
		// Do a ray cast to ensure we have a copy of the map
		map->raycast(trackingReference, blockCount);
		trackingReference->updateImagePyramid();

		// Update key frame references
		currentKeyFrame = currentFrame;
		latestTrackedFrame = currentFrame;
		// New key frames to be added to the map
		newKeyFrames.push_back(currentKeyFrame);

		return;
	}

	// Track current frame w.r.t. *LATEST TRACKED FRAME*
	SE3 poseUpdate = tracker->trackSE3(trackingReference, trackingTarget);
	// Update current frame pose
	currentFrame->pose() = latestTrackedFrame->pose() * poseUpdate.inverse();
	currentFrame->poseStruct->parentPose = latestTrackedFrame->poseStruct;

	if(tracker->trackingWasGood)
	{
		// Set last tracked frame to current frame
		latestTrackedFrame = currentFrame;

		// Do a ray cast to ensure we have a copy of the map
		int blockCount = map->fuseImages(trackingTarget);

		// Update visualisation
		updateVisualisation();

		currentFrame->poseStruct->parentPose = latestTrackedFrame->poseStruct;

		// Check if we insert a new key frame
		SE3 se3Update = currentKeyFrame->pose().inverse() * currentFrame->pose();
		Sophus::Vector3d dist = se3Update.translation();
		Sophus::Matrix3d angle = se3Update.rotationMatrix();
		Sophus::Vector3d sina = angle.eulerAngles(0, 1, 2).array().sin();
		if (dist.norm() > 0.04f || sina.norm() > 0.04f)
		{
			// Create new key frame
			currentKeyFrame = currentFrame;
			// New key frames to be added to the map
			newKeyFrames.push_back(currentKeyFrame);

			++systemState.numTrackedKeyFrames;
		}

		std::swap(trackingReference, trackingTarget);

		// full ray cast to get a copy of the map
		map->raycast(trackingReference);
		trackingReference->updateImagePyramid();
	}

	++systemState.numTrackedFrames;
}

//void SlamSystem::trackFrame(cv::Mat& img, cv::Mat& depth, int id, double timeStamp)
//{
//	// process messages before tracking
//	// this include system reboot requests
//	// and other system directives.
//	processMessages();
//
//	// create new frame and generate point cloud
//	Frame* currentFrame = new Frame(img, depth, id, K, timeStamp);
//
//	trackingTarget->generateCloud(currentFrame,false);
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
//		int numVisibleBlocks = map->fuseImages(trackingReference);
//		map->raycast(trackingReference, numVisibleBlocks);
//		trackingReference->generatePyramid();
//		latestTrackedFrame = currentFrame;
//		currentKeyFrame = currentFrame;
//		keyFrameGraph->addKeyFrame(currentKeyFrame);
//		return;
//	}
//
//	// Initial pose estimate
//	SE3 initialEstimate = latestTrackedFrame->pose().inverse() * currentKeyFrame->pose();
//
//	// Track current frame
//	// Return frame to current *Key Frame* transform
//	SE3 poseUpdate = tracker->trackSE3(trackingReference, trackingTarget, initialEstimate,false);
//
//	if(!tracker->trackingWasGood) {
//		return;
//	}
//
//	// Update current frame pose
//	currentFrame->pose() = currentKeyFrame->pose() * poseUpdate.inverse();
//
//	// Set last tracked frame to current frame
//	currentFrame->poseStruct->parentPose = currentKeyFrame->poseStruct;
//	latestTrackedFrame = currentFrame;
//
//	// Do a raycast to ensure we have a copy of the map
//	int numVisibleBlocks = map->fuseImages(trackingTarget);
//
//	// Update visualisation
//	updateVisualisation();
//
//	// Check if we insert a new key frame
//	if(tracker->trackingWasGood)
//	{
//		currentFrame->poseStruct->parentPose = currentKeyFrame->poseStruct;
//		Sophus::Vector3d dist = poseUpdate.translation();
//		Sophus::Matrix3d angle = poseUpdate.rotationMatrix();
//		Sophus::Vector3d sina = angle.eulerAngles(0, 1, 2).array().sin();
//		if(dist.norm() > 0.05f || sina.norm() > 0.05f)
//		{
//			std::swap(trackingReference, trackingTarget);
//
//			// Create new key frame
//			currentKeyFrame = currentFrame;
//
//			// New key frames to be added to the map
//			newKeyFrames.push_back(currentKeyFrame);
//
//			++systemState.numTrackedKeyFrames;
//		}
//
//
//		// full ray cast to get a copy of the map
//		map->raycast(trackingReference);
//		trackingReference->generatePyramid();
//	}
//
//	++systemState.numTrackedFrames;
//}

void SlamSystem::processMessages()
{
	std::unique_lock<std::mutex> lock(messageQueueMutex);
	while(messageQueue.size() != 0)
	{
		Msg newmsg = messageQueue.front();

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
}

void SlamSystem::queueMessage(Msg newmsg)
{
	std::unique_lock<std::mutex> lock(messageQueueMutex);
	messageQueue.push(newmsg);
}

void SlamSystem::loopMapGeneration()
{
	CONSOLE("Map Generation Thread Started.");
	std::unique_lock<std::mutex> lock(keyFramesToBeMappedMutex);
	while(keepRunning)
	{
		if(keyFramesToBeMapped.size() == 0)
		{
			lock.unlock();

			lock.lock();
		}
		else
		{
			Frame* keyFrame = keyFramesToBeMapped.front();
			if(keyFrame->keyPoints != 0)
				continue;

			keyFramesToBeMapped.pop_front();
		}
	}

	CONSOLE("Map Generation Thread Exited.");
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
		if(newConstraintAdded)
		{
			newConstraintAdded = false;
			CONSOLE("Start Optimisation.");
			while(Optimization(5, 0.02));
			keyFrameGraph->updatePoseGraph();
		}
		else
		{
			std::this_thread::__sleep_for(std::chrono::seconds(1), std::chrono::nanoseconds(0));
			CONSOLE("Optimisation Thread Waiting for Command.");
		}
	}

	CONSOLE("Optimisation Thread Started.");
}

bool SlamSystem::Optimization(int it, float minDelta)
{
	keyFrameGraph->addElementsFromBuffer();
	int its = keyFrameGraph->optimize(it);
	havePoseUpdate = true;
	return false;
}

void SlamSystem::updateMap(int nKeyFrame)
{
	std::vector<Frame*> keyframesAll = keyFrameGraph->getKeyFramesAll();
	for (auto frame : keyframesAll)
	{
		SE3 diff = frame->pose().inverse() * QuattoSE3(frame->poseStruct->graphVertex->estimate());
		frame->poseStruct->diff = diff.log().norm();
	}

	std::sort(keyframesAll.begin(), keyframesAll.end(), [](Frame* a, Frame* b) {return a->poseStruct->diff > b->poseStruct->diff;});

	for (int i = 0; i < std::min(nKeyFrame, (int) keyframesAll.size()); ++i)
	{
		// TODO: map reconstruction
		if (keyframesAll[i]->poseStruct->diff >= 0.01)
		{
			PointCloud* pc = new PointCloud();
			pc->generateCloud(keyframesAll[i]);
			map->defuseImages(pc);
			keyframesAll[i]->poseStruct->isOptimised = true;
			keyframesAll[i]->poseStruct->applyPoseUpdate();
			map->fuseImages(pc);
			delete pc;
		}
	}
}

void SlamSystem::loopConstraintSearch()
{
	CONSOLE("Constraint Search Thread Started.");
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
			findConstraintsForNewKFs(newKF);
			lock.lock();
		}
	}

	CONSOLE("Constraint Search Thread Exited.");
}

void SlamSystem::findConstraintsForNewKFs(Frame* newKF)
{
	std::unordered_set<Frame*, std::hash<Frame*>> candidates;
	std::unordered_set<Frame*, std::hash<Frame*>> closeCandidates;
	candidates = keyFrameGraph->findTrackableCandidates(newKF);
	std::map<Frame*, SE3> SE3CandidateToFrame;
	std::map<Frame*, SE3> SE3CandidateToFrameTracked;
	std::vector<KFConstraintStruct*> constraints;

	// Erase ones that are already neighbours
	for (std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	{
		if(newKF->neighbors.find(*c) != newKF->neighbors.end())
			c = candidates.erase(c);
		else
			++c;
	}

	for (Frame* candidate : candidates)
	{
		SE3 KFtoCandidate = candidate->pose().inverse() * newKF->pose();
		SE3CandidateToFrame[candidate] = KFtoCandidate;
	}

	if(candidates.size() > 0)
	{
		firstFrame->generateCloud(newKF, false);
	}

	for (Frame* candidate : candidates)
	{
		if (candidate->id() == newKF->id())
			continue;
		if (newKF->hasTrackingParent() && newKF->getTrackingParent() == candidate)
			continue;
		if (candidate->hasTrackingParent() && candidate->getTrackingParent() == newKF)
			continue;

		secondFrame->generateCloud(candidate, false);

		SE3 c2f_init = SE3CandidateToFrame[candidate];
		SE3 c2f = constraintTracker->trackSE3(firstFrame, secondFrame, c2f_init, false);
		if(!constraintTracker->trackingWasGood)
			continue;

		SE3 f2c_init = SE3CandidateToFrame[candidate].inverse();
		SE3 f2c = constraintTracker->trackSE3(secondFrame, firstFrame, f2c_init, false);
		if(!constraintTracker->trackingWasGood)
					continue;

		if((f2c * c2f).log().norm() >= 0.05)
			continue;

		CONSOLE((f2c * c2f).log().norm());
		SE3CandidateToFrameTracked[candidate] = c2f;
		closeCandidates.insert(candidate);
		newKF->neighbors.insert(candidate);
	}

	for (Frame* candidate : closeCandidates)
	{
		KFConstraintStruct* e1 = 0, * e2 = 0;
		SE3 dSE3 = SE3CandidateToFrameTracked[candidate];
		SE3 c2f = newKF->pose() * dSE3 * newKF->pose().inverse();

		e1 = new KFConstraintStruct();
		e1->first = candidate;
		e1->second = newKF;
		e1->firstToSecond = SE3toQuat(c2f);
		e1->information.setIdentity();

		e2 = new KFConstraintStruct();
		e2->first = newKF;
		e2->second = candidate;
		e2->firstToSecond = SE3toQuat(c2f.inverse());
		e2->information.setIdentity();

		if(e1 != 0 && e2 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);
		}
	}


	keyFrameGraph->addKeyFrame(newKF);
	for (KFConstraintStruct* e : constraints)
		keyFrameGraph->insertConstraint(e);

	if (newKF->hasTrackingParent())
	{
		// Insert a new constraint into the map
		KFConstraintStruct* e = new KFConstraintStruct();
		e->first = newKF->getTrackingParent();
		e->second = newKF;
		e->information.setIdentity();
		e->firstToSecond = SE3toQuat(newKF->pose() * newKF->getTrackingParent()->pose().inverse());
		keyFrameGraph->insertConstraint(e);
	}

	newConstraintAdded = true;
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
	viewer->setKeyFrameGraph(keyFrameGraph->getKeyFramesAll());

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

void SlamSystem::writeBinaryMapToDisk()
{
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

void SlamSystem::readBinaryMapFromDisk()
{
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
	keyFrameGraph->reinitialiseGraph();
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

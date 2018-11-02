#include "VoxelMap.h"
#include "Settings.h"
#include "Frame.h"
#include "PointCloud.h"
#include "DeviceFuncs.h"

VoxelMap::VoxelMap() :
		meshUpdated(false), hasNewKFFlag(false) {

}

void VoxelMap::allocateDeviceMemory()
{
	currentState.blockSize = 8;
	currentState.blockSize3 = 512;
	currentState.depthMin_raycast = 0.1f;
	currentState.depthMax_raycast = 3.0f;
	currentState.voxelSize = 0.006f;
	currentState.maxNumBuckets = 1000000;
	currentState.maxNumHashEntries = 1500000;
	currentState.maxNumVoxelBlocks = 700000;
	currentState.maxNumRenderingBlocks = 260000;
	currentState.maxNumMeshTriangles = 20000000;
	updateMapState(currentState);

	heapCounter.create(1);
	hashCounter.create(1);
	noVisibleEntries.create(1);
	heap.create(currentState.maxNumVoxelBlocks);
	sdfBlock.create(currentState.maxNumVoxels());
	bucketMutex.create(currentState.maxNumBuckets);
	hashEntries.create(currentState.maxNumHashEntries);
	visibleEntries.create(currentState.maxNumHashEntries);

	nBlocks.create(1);
	noTriangles.create(1);
	modelVertex.create(currentState.maxNumMeshVertices());
	modelNormal.create(currentState.maxNumMeshVertices());
	modelColor.create(currentState.maxNumMeshVertices());
	blockPoses.create(currentState.maxNumHashEntries);

	edgeTable.create(256);
	vertexTable.create(256);
	triangleTable.create(16, 256);
	edgeTable.upload(edgeTableHost);
	vertexTable.upload(vertexTableHost);
	triangleTable.upload(triangleTableHost);

	zRangeMin.create(80, 60);
	zRangeMax.create(80, 60);
	zRangeMinEnlarged.create(160, 120);
	zRangeMaxEnlarged.create(160, 120);
	noRenderingBlocks.create(1);
	renderingBlockList.create(currentState.maxNumRenderingBlocks);

	noKeys.create(1);
	mutexKeys.create(KeyMap::MaxKeys);
	mapKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	surfKeys.create(3000);
	mapKeyIndex.create(3000);

	CONSOLE("===============================");
	CONSOLE("VOXEL MAP successfully created.");
	CONSOLE("COUNT(HASH ENTRY) - " + std::to_string(currentState.maxNumHashEntries));
	CONSOLE("COUNT(VOXEL BLOCK) - " + std::to_string(currentState.maxNumVoxels()));
	CONSOLE("===============================");

	resetMapStruct();
}
//
//void VoxelMap::allocateDeviceMemory(MapState state) {
//
//	heapCounter.create(1);
//	hashCounter.create(1);
//	noVisibleEntries.create(1);
//	heap.create(MapStruct::NumSdfBlocks);
//	sdfBlock.create(MapStruct::NumVoxels);
//	bucketMutex.create(MapStruct::NumBuckets);
//	hashEntries.create(MapStruct::NumEntries);
//	visibleEntries.create(MapStruct::NumEntries);
//
//	nBlocks.create(1);
//	noTriangles.create(1);
//	modelVertex.create(MapStruct::MaxVertices);
//	modelNormal.create(MapStruct::MaxVertices);
//	modelColor.create(MapStruct::MaxVertices);
//	blockPoses.create(MapStruct::NumEntries);
//
//	edgeTable.create(256);
//	vertexTable.create(256);
//	triangleTable.create(16, 256);
//	edgeTable.upload(edgeTableHost);
//	vertexTable.upload(vertexTableHost);
//	triangleTable.upload(triangleTableHost);
//
//	zRangeMin.create(80, 60);
//	zRangeMax.create(80, 60);
//	zRangeMinEnlarged.create(160, 120);
//	zRangeMaxEnlarged.create(160, 120);
//	noRenderingBlocks.create(1);
//	renderingBlockList.create(MapStruct::MaxRenderingBlocks);
//
//	noKeys.create(1);
//	mutexKeys.create(KeyMap::MaxKeys);
//	mapKeys.create(KeyMap::maxEntries);
//	tmpKeys.create(KeyMap::maxEntries);
//	surfKeys.create(3000);
//	mapKeyIndex.create(3000);
//
//	CONSOLE("===============================");
//	CONSOLE("VOXEL MAP successfully created.");
//	CONSOLE("COUNT(HASH ENTRY) - " + std::to_string(MapStruct::NumEntries));
//	CONSOLE("COUNT(VOXEL BLOCK) - " + std::to_string(MapStruct::NumVoxels));
//	CONSOLE("===============================");
//	resetMapStruct();
//}

void VoxelMap::ForwardWarp(Frame * last, Frame * next) {
	ForwardWarping(last->vmap[0], last->nmap[0], next->vmap[0], next->nmap[0],
			last->GpuRotation(), next->GpuInvRotation(), last->GpuTranslation(),
			next->GpuTranslation(), Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0));
}

//void VoxelMap::UpdateVisibility(const KeyFrame * kf, uint & no) {
//
////	CheckBlockVisibility(*this, noVisibleEntries, kf->GpuRotation(),
////			kf->GpuInvRotation(), kf->GpuTranslation(), Frame::cols(0),
////			Frame::rows(0), Frame::fx(0), Frame::fy(0), Frame::cx(0),
////			Frame::cy(0), DeviceMap::DepthMax, DeviceMap::DepthMin, &no);
//}

void VoxelMap::UpdateVisibility(Frame * f, uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, f->GpuRotation(), f->GpuInvRotation(),
			f->GpuTranslation(), Frame::cols(0), Frame::rows(0), Frame::fx(0),
			Frame::fy(0), Frame::cx(0), Frame::cy(0), currentState.depthMin_raycast,
			currentState.depthMax_raycast, &no);
}

void VoxelMap::UpdateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy,
		uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, Rview, RviewInv, tview, 640,
			480, fx, fy, cx, cy, depthMax, depthMin, &no);
}

void VoxelMap::FuseColor(Frame * f, uint & no) {
	FuseColor(f->range, f->color, f->nmap[0], f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation(), no);
}

void VoxelMap::DefuseColor(Frame * f, uint & no) {
	FuseColor(f->range, f->color, f->nmap[0], f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation(), no);
}

void VoxelMap::DefuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no) {

	DefuseMapColor(depth, color, normal, noVisibleEntries, Rview, RviewInv,
			tview, *this, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0), currentState.depthMin_raycast,
			currentState.depthMax_raycast, &no);

}

void VoxelMap::FuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no) {

	FuseMapColor(depth, color, normal, noVisibleEntries, Rview, RviewInv, tview,
			*this, Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			currentState.depthMin_raycast, currentState.depthMax_raycast, &no);

}

void VoxelMap::RayTrace(uint noVisibleBlocks, Frame * f) {
	RayTrace(noVisibleBlocks, f->GpuRotation(), f->GpuInvRotation(),
			f->GpuTranslation(), f->vmap[0], f->nmap[0],
			currentState.depthMin_raycast, currentState.depthMax_raycast,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0));
}

void VoxelMap::RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float4> & nmap,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy) {

	if (CreateRenderingBlocks(visibleEntries, zRangeMin, zRangeMax, depthMax, depthMin,
			renderingBlockList, noRenderingBlocks, RviewInv, tview,
			noVisibleBlocks, fx, fy, cx, cy)) {

		Raycast(*this, vmap, nmap, zRangeMin, zRangeMax, Rview, RviewInv, tview,
				1.0 / fx, 1.0 / fy, cx, cy);
	}
}

void VoxelMap::CreateModel() {

	MeshScene(nBlocks, noTriangles, *this, edgeTable, vertexTable,
			triangleTable, modelNormal, modelVertex, modelColor, blockPoses);

	noTriangles.download(&noTrianglesHost);
	if (noTrianglesHost > 0) {
		meshUpdated = true;
	}
}

void VoxelMap::UpdateMapKeys() {
	noKeys.clear();
	CollectKeyPoints(*this, tmpKeys, noKeys);

	noKeys.download(&noKeysHost);
	if(noKeysHost != 0) {
		hostKeys.resize(noKeysHost);
		tmpKeys.download(hostKeys.data(), noKeysHost);
	}
}

void VoxelMap::CreateRAM() {

//	downloadMapState(currentState);
//
//	heapCounterRAM = new int[1];
//	hashCounterRAM = new int[1];
//	noVisibleEntriesRAM = new uint[1];
//	heapRAM = new int[MapStruct::NumSdfBlocks];
//	bucketMutexRAM = new int[MapStruct::NumBuckets];
//	sdfBlockRAM = new Voxel[MapStruct::NumVoxels];
//	hashEntriesRAM = new HashEntry[MapStruct::NumEntries];
//	visibleEntriesRAM = new HashEntry[MapStruct::NumEntries];
//
//	mutexKeysRAM = new int[KeyMap::MaxKeys];
//	mapKeysRAM = new SURF[KeyMap::maxEntries];
}

void VoxelMap::DownloadToRAM() {

	CreateRAM();

	heapCounter.download(heapCounterRAM);
	hashCounter.download(hashCounterRAM);
	noVisibleEntries.download(noVisibleEntriesRAM);
	heap.download(heapRAM);
	bucketMutex.download(bucketMutexRAM);
	sdfBlock.download(sdfBlockRAM);
	hashEntries.download(hashEntriesRAM);
	visibleEntries.download(visibleEntriesRAM);

	mutexKeys.download(mutexKeysRAM);
	mapKeys.download(mapKeysRAM);
}

void VoxelMap::UploadFromRAM() {

	heapCounter.upload(heapCounterRAM);
	hashCounter.upload(hashCounterRAM);
	noVisibleEntries.upload(noVisibleEntriesRAM);
	heap.upload(heapRAM);
	bucketMutex.upload(bucketMutexRAM);
	sdfBlock.upload(sdfBlockRAM);
	hashEntries.upload(hashEntriesRAM);
	visibleEntries.upload(visibleEntriesRAM);

	mutexKeys.upload(mutexKeysRAM);
	mapKeys.upload(mapKeysRAM);
}

void VoxelMap::ReleaseRAM() {

	delete [] heapCounterRAM;
	delete [] hashCounterRAM;
	delete [] noVisibleEntriesRAM;
	delete [] heapRAM;
	delete [] bucketMutexRAM;
	delete [] sdfBlockRAM;
	delete [] hashEntriesRAM;
	delete [] visibleEntriesRAM;

	delete [] mutexKeysRAM;
	delete [] mapKeysRAM;
}

bool VoxelMap::HasNewKF() {

	return hasNewKFFlag;
}

//void VoxelMap::FuseKeyFrame(const KeyFrame * kf) {

//	if (keyFrames.count(kf))
//		return;
//
//	keyFrames.insert(kf);
//
//	cv::Mat desc;
//	std::vector<int> index;
//	std::vector<int> keyIndex;
//	std::vector<SURF> keyChain;
//	kf->descriptors.download(desc);
//	kf->outliers.resize(kf->N);
//	std::fill(kf->outliers.begin(), kf->outliers.end(), true);
//	int noK = std::min(kf->N, (int) surfKeys.size);
//
//	kf->pt3d.resize(kf->N);
//
//	for (int i = 0; i < noK; ++i) {
//
//		if (kf->observations[i] > 0) {
//			SURF key;
//			Eigen::Vector3f pt = kf->GetWorldPoint(i);
//			key.pos = {pt(0), pt(1), pt(2)};
//			key.normal = kf->pointNormal[i];
//			key.valid = true;
//
//			for (int j = 0; j < 64; ++j) {
//				key.descriptor[j] = desc.at<float>(i, j);
//			}
//
//			index.push_back(i);
//			keyChain.push_back(key);
//			keyIndex.push_back(kf->keyIndex[i]);
//			kf->outliers[i] = false;
//		}
//	}
//
//	std::cout << "Num KP fused : " << std::count(kf->outliers.begin(), kf->outliers.end(), false) << std::endl;
//
//	surfKeys.upload(keyChain.data(), keyChain.size());
//	mapKeyIndex.upload(keyIndex.data(), keyIndex.size());
//
//	InsertKeyPoints(*this, surfKeys, mapKeyIndex, keyChain.size());
//
//	mapKeyIndex.download(keyIndex.data(), keyIndex.size());
//	surfKeys.download(keyChain.data(), keyChain.size());
//
//	for(int i = 0; i < index.size(); ++i) {
//		int idx = index[i];
//		kf->keyIndex[idx] = keyIndex[i];
//		float3 pos = keyChain[i].pos;
//		kf->mapPoints[idx] << pos.x, pos.y, pos.z;
//	}
//
//	if(localMap.size() > 0) {
//		if(localMap.size() >= 7) {
//			localMap.erase(localMap.begin());
//			localMap.push_back(kf);
//		}
//		else
//			localMap.push_back(kf);
//	}
//	else
//		localMap.push_back(kf);
//
//	newKF = const_cast<KeyFrame *>(kf);
//
//	FindLocalGraph(newKF);
//	poseGraph.insert(newKF);
//
//	hasNewKFFlag = true;
//}

//void VoxelMap::FindLocalGraph(KeyFrame * kf) {
//
////	const float distTH = 0.5f;
////	const float angleTH = 0.3f;
////	Eigen::Vector3f viewDir = kf->Rotation().rightCols<1>();
////
////	std::vector<KeyFrame *> kfCandidates;
////
////	for(std::set<KeyFrame *>::iterator iter = poseGraph.begin(),lend = poseGraph.end();	iter != lend; ++iter) {
////
////		KeyFrame * candidate = *iter;
////
////		if(candidate->frameId == kf->frameId)
////			continue;
////
////		float dist = candidate->Translation().dot(kf->Translation());
////		if(dist > distTH)
////			continue;
////
////		Eigen::Vector3f dir = candidate->Rotation().rightCols<1>();
////		float angle = viewDir.dot(dir);
////
////		if(angle < angleTH)
////			continue;
////
////		kfCandidates.push_back(candidate);
////		std::cout << angle << "/" << angleTH << "  " << dist << "/" << distTH << std::endl;
////	}
//}
//
//void VoxelMap::FuseKeyPoints(Frame * f) {
//
//	std::cout << "NOT IMPLEMENTED" << std::endl;
//}

void VoxelMap::resetMapStruct()
{

	CONSOLE("Re-Initialising Device Memory.");

	ResetMap(*this);
	ResetKeyPoints(*this);

	mapKeys.clear();
}

VoxelMap::operator KeyMap() const {

	KeyMap map;

	map.Keys = mapKeys;
	map.Mutex = mutexKeys;

	return map;
}

VoxelMap::operator MapStruct() const {

	MapStruct map;

	map.heapMem = heap;
	map.heapCounter = heapCounter;
	map.noVisibleBlocks = noVisibleEntries;
	map.bucketMutex = bucketMutex;
	map.hashEntries = hashEntries;
	map.visibleEntries = visibleEntries;
	map.voxelBlocks = sdfBlock;
	map.entryPtr = hashCounter;

	return map;
}

//======================== REFACOTRING ========================

VoxelMap::~VoxelMap()
{

}

void VoxelMap::writeMapToDisk(std::string path)
{

}

void VoxelMap::readMapFromDisk(std::string path)
{

}

void VoxelMap::copyMapDeviceToHost()
{

}

void VoxelMap::copyMapHostToDevice()
{

}

void VoxelMap::allocateHostMemory()
{

}

void VoxelMap::releaseHostMemory()
{

}

void VoxelMap::releaseDeviceMemory()
{

}

int VoxelMap::fusePointCloud(PointCloud* data)
{
	uint no;
	Frame* trackingFrame = data->frame;
	FuseMapColor(data->depth_float,
			data->image_raw,
			data->nmap[0],
			noVisibleEntries,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			*this,
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy(),
			currentState.depthMax_raycast,
			currentState.depthMin_raycast,
			&no);
	return (int)no;
}

void VoxelMap::takeSnapShot(PointCloud* data, int numVisibleBlocks)
{
	Frame* trackingFrame = data->frame;
	uint no = numVisibleBlocks < 0 ? updateVisibility(data) : (uint) numVisibleBlocks;
	RayTrace(no,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			data->vmap[0],
			data->nmap[0],
			currentState.depthMin_raycast,
			currentState.depthMax_raycast,
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy());
}

uint VoxelMap::updateVisibility(PointCloud* data)
{
	uint no(0);
	Frame* trackingFrame = data->frame;
	CheckBlockVisibility(*this,
			noVisibleEntries,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			trackingFrame->width(),
			trackingFrame->height(),
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy(),
			currentState.depthMax_raycast,
			currentState.depthMin_raycast,
			&no);
	return no;
}

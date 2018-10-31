#include "DenseMap.h"
#include "Constant.h"
#include "PointCloud.h"
#include "DeviceFuncs.h"

VoxelMap::VoxelMap() :
		meshUpdated(false), hasNewKFFlag(false) {
	allocateDeviceMemory();
}

void VoxelMap::allocateDeviceMemory() {

	heapCounter.create(1);
	hashCounter.create(1);
	noVisibleEntries.create(1);
	heap.create(DeviceMap::NumSdfBlocks);
	sdfBlock.create(DeviceMap::NumVoxels);
	bucketMutex.create(DeviceMap::NumBuckets);
	hashEntries.create(DeviceMap::NumEntries);
	visibleEntries.create(DeviceMap::NumEntries);

	nBlocks.create(1);
	noTriangles.create(1);
	modelVertex.create(DeviceMap::MaxVertices);
	modelNormal.create(DeviceMap::MaxVertices);
	modelColor.create(DeviceMap::MaxVertices);
	blockPoses.create(DeviceMap::NumEntries);

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
	renderingBlockList.create(DeviceMap::MaxRenderingBlocks);

	noKeys.create(1);
	mutexKeys.create(KeyMap::MaxKeys);
	mapKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	surfKeys.create(3000);
	mapKeyIndex.create(3000);

	Reset();
}

void VoxelMap::ForwardWarp(Frame * last, Frame * next) {
	ForwardWarping(last->vmap[0], last->nmap[0], next->vmap[0], next->nmap[0],
			last->GpuRotation(), next->GpuInvRotation(), last->GpuTranslation(),
			next->GpuTranslation(), Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0));
}

void VoxelMap::UpdateVisibility(const KeyFrame * kf, uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, kf->GpuRotation(),
			kf->GpuInvRotation(), kf->GpuTranslation(), Frame::cols(0),
			Frame::rows(0), Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0), DeviceMap::DepthMax, DeviceMap::DepthMin, &no);
}

void VoxelMap::UpdateVisibility(Frame * f, uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, f->GpuRotation(), f->GpuInvRotation(),
			f->GpuTranslation(), Frame::cols(0), Frame::rows(0), Frame::fx(0),
			Frame::fy(0), Frame::cx(0), Frame::cy(0), DeviceMap::DepthMax,
			DeviceMap::DepthMin, &no);
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

	DefuseMapColor(depth, color, normal, noVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin, &no);

}

void VoxelMap::FuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no) {

	FuseMapColor(depth, color, normal, noVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin, &no);

}

void VoxelMap::RayTrace(uint noVisibleBlocks, Frame * f) {
	RayTrace(noVisibleBlocks, f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation(),
			f->vmap[0], f->nmap[0], DeviceMap::DepthMin, DeviceMap::DepthMax,
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

	heapCounterRAM = new int[1];
	hashCounterRAM = new int[1];
	noVisibleEntriesRAM = new uint[1];
	heapRAM = new int[DeviceMap::NumSdfBlocks];
	bucketMutexRAM = new int[DeviceMap::NumBuckets];
	sdfBlockRAM = new Voxel[DeviceMap::NumVoxels];
	hashEntriesRAM = new HashEntry[DeviceMap::NumEntries];
	visibleEntriesRAM = new HashEntry[DeviceMap::NumEntries];

	mutexKeysRAM = new int[KeyMap::MaxKeys];
	mapKeysRAM = new SURF[KeyMap::maxEntries];
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

void VoxelMap::FuseKeyFrame(const KeyFrame * kf) {

	if (keyFrames.count(kf))
		return;

	keyFrames.insert(kf);

	cv::Mat desc;
	std::vector<int> index;
	std::vector<int> keyIndex;
	std::vector<SURF> keyChain;
	kf->descriptors.download(desc);
	kf->outliers.resize(kf->N);
	std::fill(kf->outliers.begin(), kf->outliers.end(), true);
	int noK = std::min(kf->N, (int) surfKeys.size);

	kf->pt3d.resize(kf->N);

	for (int i = 0; i < noK; ++i) {

		if (kf->observations[i] > 0) {
			SURF key;
			Eigen::Vector3f pt = kf->GetWorldPoint(i);
			key.pos = {pt(0), pt(1), pt(2)};
			key.normal = kf->pointNormal[i];
			key.valid = true;

			for (int j = 0; j < 64; ++j) {
				key.descriptor[j] = desc.at<float>(i, j);
			}

			index.push_back(i);
			keyChain.push_back(key);
			keyIndex.push_back(kf->keyIndex[i]);
			kf->outliers[i] = false;
		}
	}

	std::cout << "Num KP fused : " << std::count(kf->outliers.begin(), kf->outliers.end(), false) << std::endl;

	surfKeys.upload(keyChain.data(), keyChain.size());
	mapKeyIndex.upload(keyIndex.data(), keyIndex.size());

	InsertKeyPoints(*this, surfKeys, mapKeyIndex, keyChain.size());

	mapKeyIndex.download(keyIndex.data(), keyIndex.size());
	surfKeys.download(keyChain.data(), keyChain.size());

	for(int i = 0; i < index.size(); ++i) {
		int idx = index[i];
		kf->keyIndex[idx] = keyIndex[i];
		float3 pos = keyChain[i].pos;
		kf->mapPoints[idx] << pos.x, pos.y, pos.z;
	}

	if(localMap.size() > 0) {
		if(localMap.size() >= 7) {
			localMap.erase(localMap.begin());
			localMap.push_back(kf);
		}
		else
			localMap.push_back(kf);
	}
	else
		localMap.push_back(kf);

	newKF = const_cast<KeyFrame *>(kf);

	FindLocalGraph(newKF);
	poseGraph.insert(newKF);

	hasNewKFFlag = true;
}

void VoxelMap::FindLocalGraph(KeyFrame * kf) {

	const float distTH = 0.5f;
	const float angleTH = 0.3f;
	Eigen::Vector3f viewDir = kf->Rotation().rightCols<1>();

	std::vector<KeyFrame *> kfCandidates;

	for(std::set<KeyFrame *>::iterator iter = poseGraph.begin(),lend = poseGraph.end();	iter != lend; ++iter) {

		KeyFrame * candidate = *iter;

		if(candidate->frameId == kf->frameId)
			continue;

		float dist = candidate->Translation().dot(kf->Translation());
		if(dist > distTH)
			continue;

		Eigen::Vector3f dir = candidate->Rotation().rightCols<1>();
		float angle = viewDir.dot(dir);

		if(angle < angleTH)
			continue;

		kfCandidates.push_back(candidate);
		std::cout << angle << "/" << angleTH << "  " << dist << "/" << distTH << std::endl;
	}
}

void VoxelMap::FuseKeyPoints(Frame * f) {

	std::cout << "NOT IMPLEMENTED" << std::endl;
}

void VoxelMap::Reset() {

	ResetMap(*this);
	ResetKeyPoints(*this);

	mapKeys.clear();
	keyFrames.clear();
}

VoxelMap::operator KeyMap() const {

	KeyMap map;

	map.Keys = mapKeys;
	map.Mutex = mutexKeys;

	return map;
}

VoxelMap::operator DeviceMap() const {

	DeviceMap map;

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

VoxelMap::VoxelMap(float voxelSize, int numSdfBlock, int numHashEntry)
{
	if(numHashEntry < numSdfBlock)
	{
		printf("ERROR creating the map.\n");
		return;
	}

	param.voxelSize = voxelSize;
	param.numSdfBlock = numSdfBlock;
	param.numEntries = numHashEntry;
	param.numBuckets = (uint)((float)numHashEntry * 0.75);
	param.numExcessBlock = numHashEntry - param.numBuckets;
	param.maxNumRenderingBlock = 260000;
	param.numVoxels = numSdfBlock * param.blockSize3;
	param.maxNumTriangles = 20000000; // that's roughly 700MB of memory
	param.maxNumVertices = param.maxNumTriangles * 3;
	param.truncateDist = voxelSize * 8;
}

VoxelMap::~VoxelMap()
{
	if(device_map.memoryAllocated)
		releaseDeviceMemory();

	if(host_map.memoryAllocated)
		releaseHostMemory();
}

void VoxelMap::writeMapToDisk(std::string path)
{
	if(!host_map.memoryAllocated)
		allocateHostMemory();

	copyMapDeviceToHost();

	auto file = std::fstream(path, std::ios::out | std::ios::binary);

	// begin writing of general map info
	file.write((const char*) &param.numSdfBlock, sizeof(int));
	file.write((const char*) &param.numBuckets, sizeof(int));
	file.write((const char*) &param.numVoxels, sizeof(int));
	file.write((const char*) &param.numEntries, sizeof(int));

	// begin writing of dense map
	file.write((char*) host_map.ptr_heap, sizeof(int));
	file.write((char*) host_map.ptr_entry, sizeof(int));
	file.write((char*) host_map.num_visible_blocks, sizeof(uint));
	file.write((char*) host_map.heap, sizeof(int) * param.numSdfBlock);
	file.write((char*) host_map.mutex_bucket, sizeof(int) * param.numBuckets);
	file.write((char*) host_map.sdf_blocks, sizeof(Voxel) * param.numVoxels);
	file.write((char*) host_map.hash_table, sizeof(HashEntry) * param.numEntries);
	file.write((char*) host_map.visible_entry, sizeof(HashEntry) * param.numEntries);

	// clean up
	file.close();

	releaseHostMemory();
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
	if(host_map.memoryAllocated)
	{
		printf("Host memory IS already allocated, Exit.\n");
		return;
	}
	else
	{
		host_map.ptr_entry = new int[1];
		host_map.ptr_heap = new int[1];
		host_map.heap = new int[param.numSdfBlock];
		host_map.hash_table = new HashEntry[param.numEntries];
		host_map.visible_entry = new HashEntry[param.numEntries];
		host_map.mutex_bucket = new int[param.numBuckets];

		host_map.memoryAllocated = true;
	}
}

void VoxelMap::releaseHostMemory()
{
	if(host_map.memoryAllocated)
	{
		delete host_map.ptr_entry;
		delete host_map.ptr_heap;
		delete host_map.heap;
		delete host_map.hash_table;
		delete host_map.visible_entry;
		delete host_map.mutex_bucket;
		host_map.memoryAllocated = false;
	}
}

void VoxelMap::releaseDeviceMemory()
{
	if(device_map.memoryAllocated)
	{
		cudaFree(device_map.ptr_entry);
		cudaFree(device_map.ptr_heap);
		cudaFree(device_map.heap);
		cudaFree(device_map.hash_table);
		cudaFree(device_map.visible_entry);
		cudaFree(device_map.mutex_bucket);
		device_map.memoryAllocated = false;
	}
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
			DeviceMap::DepthMax,
			DeviceMap::DepthMin,
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
			DeviceMap::DepthMin,
			DeviceMap::DepthMax,
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
			DeviceMap::DepthMax,
			DeviceMap::DepthMin,
			&no);
	return no;
}

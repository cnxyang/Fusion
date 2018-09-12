#include "map.h"
#include "timer.h"
#include "const.h"
#include "rendering.h"
#include "reduction.h"

Mapping::Mapping(bool default_stream, cudaStream_t * stream_) :
		meshUpdated(false), mapKeyUpdated(false), noKeysInMap(0),
		useDefaultStream(default_stream), stream(stream_) {
	create();
}

void Mapping::create() {

	// reconstruction
	heapCounter.create(1);
	hashCounter.create(1);
	noVisibleEntries.create(1);
	heap.create(DeviceMap::NumSdfBlocks);
	sdfBlock.create(DeviceMap::NumVoxels);
	bucketMutex.create(DeviceMap::NumBuckets);
	hashEntries.create(DeviceMap::NumEntries);
	visibleEntries.create(DeviceMap::NumEntries);

	// extraction
	nBlocks.create(1);
	noTriangles.create(1);
	modelVertex.create(DeviceMap::MaxVertices);
	modelNormal.create(DeviceMap::MaxVertices);
	modelColor.create(DeviceMap::MaxVertices);
	blockPoses.create(DeviceMap::NumEntries);
	edgeTable.create(256);
	vertexTable.create(256);
	triangleTable.create(16, 256);
	edgeTable.upload(edgeTable_host, 256);
	vertexTable.upload(numVertsTable, 256);
	triangleTable.upload(triTable_host, sizeof(int) * 16, 16, 256);

	// rendering
	zRangeMin.create(80, 60);
	zRangeMax.create(80, 60);
	noRenderingBlocks.create(1);
	renderingBlockList.create(DeviceMap::MaxRenderingBlocks);

	// key point
	mKeyMutex.create(KeyMap::MaxKeys);
	mORBKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	keyIndices.create(1500);

	reset();
}

void Mapping::createModel() {

	meshScene(nBlocks, noTriangles, *this, edgeTable, vertexTable,
			triangleTable, modelNormal, modelVertex, modelColor, blockPoses);

	noTriangles.download(&noTrianglesHost);
	if (noTrianglesHost > 0) {
		mutexMesh.lock();
		meshUpdated = true;
		mutexMesh.unlock();
	}
}

void Mapping::fuseKeys(Frame & f, std::vector<bool> & outliers) {
	std::vector<ORBKey> newKeys;
	cv::Mat descriptors;
	f.descriptors.download(descriptors);
	for(int i = 0; i < f.N; ++i) {
		ORBKey key;
		key.obs = 1;
		key.valid = true;
		cv::Vec3f normal = f.mNormals[i];
		Eigen::Vector3d worldPos = f.Rotation() * f.mPoints[i] + f.Translation();
		key.pos = make_float3((float)worldPos(0), (float)worldPos(1), (float)worldPos(2));
		key.normal = make_float3(normal(0), normal(1), normal(2));
		for(int j = 0; j < 32; ++j) {
			key.descriptor[j] = descriptors.at<char>(i, j);
		}
		newKeys.push_back(key);
	}

	DeviceArray<ORBKey> dKeys(newKeys.size());
	dKeys.upload(newKeys.data(), newKeys.size());
	InsertKeys(*this, dKeys, keyIndices);
}

std::vector<ORBKey> Mapping::getAllKeys() {
	return hostKeys;
}

void Mapping::updateMapKeys() {
	CollectKeys(*this, tmpKeys, noKeysInMap);
	if(noKeysInMap == 0)
		return;
	hostKeys.resize(noKeysInMap);
	tmpKeys.download(hostKeys.data(), noKeysInMap);
}

void Mapping::updateKeyIndices() {

}

void Mapping::fuseColor(const DeviceArray2D<float> & depth,
					    const DeviceArray2D<uchar3> & color,
					    Matrix3f Rview,
					    Matrix3f RviewInv,
					    float3 tview,
					    uint & no) {

	integrateColor(depth, color, noVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin, &no);

}

void Mapping::rayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float4> & nmap) {

	if (createRenderingBlock(visibleEntries, zRangeMin, zRangeMax, 3.0, 0.1,
			renderingBlockList, noRenderingBlocks, RviewInv, tview,
			noVisibleBlocks, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0))) {

		rayCast(*this, vmap, nmap, zRangeMin, zRangeMax, Rview, RviewInv, tview,
				1.0 / Frame::fx(0), 1.0 / Frame::fy(0), Frame::cx(0),
				Frame::cy(0));
	}
}

void Mapping::reset() {
	resetDeviceMap(*this);
	resetKeyMap(*this);
}

Mapping::operator KeyMap() {

	KeyMap map;
	map.Keys = mORBKeys;
	map.Mutex = mKeyMutex;
	return map;
}

Mapping::operator KeyMap() const {

	KeyMap map;
	map.Keys = mORBKeys;
	map.Mutex = mKeyMutex;
	return map;
}

Mapping::operator DeviceMap() {

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

Mapping::operator DeviceMap() const {

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
void Mapping::push_back(KeyFrame * kf) {
	keyFrames.insert(kf);
	std::cout << keyFrames.size() << std::endl;

	std::vector<ORBKey> newKeys;
	cv::Mat descriptors;
	kf->frameDescriptors.download(descriptors);
	for(int i = 0; i < kf->N; ++i) {
		ORBKey key;
		key.obs = 1;
		key.valid = true;
		Eigen::Vector3d worldPos = kf->rotation() * kf->frameKeys[i] + kf->translation();
		key.pos = make_float3((float)worldPos(0), (float)worldPos(1), (float)worldPos(2));
		for(int j = 0; j < 32; ++j) {
			key.descriptor[j] = descriptors.at<char>(i, j);
		}
		newKeys.push_back(key);
	}

	DeviceArray<ORBKey> dKeys(newKeys.size());
	dKeys.upload(newKeys.data(), newKeys.size());
	InsertKeys(*this, dKeys, keyIndices);
	kf->keyIndices.resize(kf->N);
	keyIndices.download((void*) kf->keyIndices.data(), kf->N);
	for(int i = 0; i < kf->N; ++i)
		std::cout << kf->keyIndices[i] << std::endl;
}

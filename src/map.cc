#include "Timer.hpp"
#include "Mapping.hpp"
#include "Table.hpp"
#include "rendering.h"
#include "reduction.h"

Mapping::Mapping() :
		meshUpdated(false) {
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
	noTriangles.create(1, true);
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
	noRenderingBlocks.create(1, true);
	renderingBlockList.create(DeviceMap::MaxRenderingBlocks);

	// key point
	mKeyMutex.create(KeyMap::MaxKeys);
//	mapPoints.create(KeyMap::MaxKeys);
	mORBKeys.create(KeyMap::MaxKeys * KeyMap::nBuckets);

	reset();
}

void Mapping::createModel() {

	meshScene(nBlocks, noTriangles, *this, edgeTable, vertexTable,
			triangleTable, modelNormal, modelVertex, modelColor, blockPoses);

	noTrianglesHost = noTriangles[0];
	if (noTrianglesHost > 0) {
		mutexMesh.lock();
		meshUpdated = true;
		mutexMesh.unlock();
	}
}

void Mapping::IntegrateKeys(Frame& F) {

	std::vector<ORBKey> keys;
	cv::Mat desc;
	F.descriptors.download(desc);
	std::cout << F.N << std::endl;
	for (int i = 0; i < F.N; ++i) {
			ORBKey key;
			key.obs = 1;
			key.valid = true;
			cv::Vec3f normal = F.mNormals[i];
			Eigen::Vector3d worldPos = F.Rotation() * F.mPoints[i] + F.Translation();
			key.pos = make_float3((float)worldPos(0), (float)worldPos(1), (float)worldPos(2));
			key.normal = make_float3(normal(0), normal(1), normal(2));
			for (int j = 0; j < 32; ++j)
				key.descriptor[j] = desc.at<char>(i, j);
			keys.push_back(key);
	}

	DeviceArray<ORBKey> dKeys(keys.size());
	dKeys.upload((void*) keys.data(), keys.size());

	InsertKeys(*this, dKeys);
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

void Mapping::rayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float3> & nmap) {

	if (createRenderingBlock(visibleEntries, zRangeMin, zRangeMax, 3.0, 0.1,
			renderingBlockList, noRenderingBlocks, RviewInv, tview,
			noVisibleBlocks, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0))) {

		rayCast(*this, vmap, nmap, zRangeMin, zRangeMax, Rview, RviewInv, tview,
				1.0 / Frame::fx(0), 1.0 / Frame::fy(0), Frame::cx(0),
				Frame::cy(0));
	}
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

void Mapping::reset() {
	resetDeviceMap(*this);
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

void Mapping::push_back(const KeyFrame * kf) {

}

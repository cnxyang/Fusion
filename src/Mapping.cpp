#include "Timer.hpp"
#include "Mapping.hpp"
#include "Table.hpp"
#include "rendering.h"
#include "reduction.h"

bool Mapping::mbFirstCall = true;

Mapping::Mapping():
nTriangle(0), bUpdated(false), extractColor(false) {}

Mapping::~Mapping() {
	release();
}

void Mapping::allocate() {

	Timer::Start("Initialisation", "Memory Allocation");
	mMemory.create(DeviceMap::NumSdfBlocks);
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(DeviceMap::NumBuckets);
	mHashEntries.create(DeviceMap::NumEntries);
	mVisibleEntries.create(DeviceMap::NumEntries);
	mVoxelBlocks.create(DeviceMap::NumSdfBlocks * DeviceMap::BlockSize3);
	mEntryPtr.create(1);

	mKeyMutex.create(KeyMap::MaxKeys);
	mORBKeys.create(KeyMap::MaxKeys * KeyMap::nBuckets);

	zRange.create(640 / 8, 480 / 8);
	mMesh.create(DeviceMap::MaxTriangles * 3);
	mMeshNormal.create(DeviceMap::MaxTriangles * 3);
	mColorMap.create(DeviceMap::MaxTriangles * 3);
	mTriTable.create(16, 256);
	mEdgeTable.create(256);
	mNoVertex.create(256);
	mTriTable.upload(triTable, sizeof(int) * 16, 16, 256);
	mEdgeTable.upload(edgeTable, 256);
	mNoVertex.upload(numVertsTable, 256);
	extractedPoses.create(DeviceMap::NumEntries);
	nBlocks.create(1);
	nTriangles.create(1);
	mRenderingBlockList.create(DeviceMap::MaxRenderingBlocks);
	mDepthMapMin.create(80, 60);
	mDepthMapMax.create(80, 60);

	reset();
}

void Mapping::release() {
	mUsedMem.release();
	mNumVisibleEntries.release();
	mBucketMutex.release();
	mMemory.release();
	mHashEntries.release();
	mVisibleEntries.release();
	mVoxelBlocks.release();
	mORBKeys.release();
	mEntryPtr.release();
}

void Mapping::createMesh() {

	nTriangle = meshScene(nBlocks, nTriangles, *this, mEdgeTable,
			mNoVertex, mTriTable, mMeshNormal, mMesh, mColorMap,
			extractedPoses);

	if(nTriangle > 0) {
		mMutexMesh.lock();
		bUpdated = true;
		mMutexMesh.unlock();
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

	integrateColor(depth, color, mNumVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin, &no);

}

Mapping::operator DeviceMap() {
	DeviceMap map;
	map.heapMem = mMemory;
	map.heapCounter = mUsedMem;
	map.noVisibleBlocks = mNumVisibleEntries;
	map.bucketMutex = mBucketMutex;
	map.hashEntries = mHashEntries;
	map.visibleEntries = mVisibleEntries;
	map.voxelBlocks = mVoxelBlocks;
	map.entryPtr = mEntryPtr;
	return map;
}

void Mapping::RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float3> & nmap) {

	DeviceArray<uint> noRenderingBlocks(1);
	Timer::Start("test", "test");
	if (createRenderingBlock(mVisibleEntries, mDepthMapMin, mDepthMapMax, 5.0, 0.1,
			mRenderingBlockList, noRenderingBlocks, RviewInv, tview,
			noVisibleBlocks, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0))) {

		rayCast(*this, vmap, nmap, mDepthMapMin, mDepthMapMax, Rview, RviewInv, tview,
				1.0 / Frame::fx(0), 1.0 / Frame::fy(0), Frame::cx(0),
				Frame::cy(0));
	}
	Timer::Stop("test", "test");

}

Mapping::operator const DeviceMap() const {
	DeviceMap map;
	map.heapMem = mMemory;
	map.heapCounter = mUsedMem;
	map.noVisibleBlocks = mNumVisibleEntries;
	map.bucketMutex = mBucketMutex;
	map.hashEntries = mHashEntries;
	map.visibleEntries = mVisibleEntries;
	map.voxelBlocks = mVoxelBlocks;
	map.entryPtr = mEntryPtr;
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

Mapping::operator const KeyMap() const {
	KeyMap map;
	map.Keys = mORBKeys;
	map.Mutex = mKeyMutex;
	return map;
}

void Mapping::push_back(const KeyFrame * kf) {

}

#include "Timer.hpp"
#include "Mapping.hpp"
#include "device_mapping.cuh"

bool Mapping::mbFirstCall = true;

Mapping::Mapping() {
}

Mapping::~Mapping() {
	ReleaseDeviceMemory();
}

void Mapping::AllocateDeviceMemory(MapDesc desc) {

	Timer::StartTiming("Initialisation", "Memory Allocation");
	mMemory.create(NUM_SDF_BLOCKS);
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(NUM_BUCKETS);
	mHashEntries.create(NUM_BUCKETS * BUCKET_SIZE);
	mVisibleEntries.create(NUM_BUCKETS * BUCKET_SIZE);
	mVoxelBlocks.create(NUM_SDF_BLOCKS * BLOCK_SIZE);

	mKeyMutex.create(KeyMap::MaxKeys);
	mORBKeys.create(KeyMap::MaxKeys * KeyMap::nBuckets);
	Timer::StopTiming("Initialisation", "Memory Allocation");

	Timer::StartTiming("Initialisation", "ResetMap");
	ResetDeviceMemory();
	Timer::StopTiming("Initialisation", "ResetMap");
}

void Mapping::ReleaseDeviceMemory() {
	mUsedMem.release();
	mNumVisibleEntries.release();
	mBucketMutex.release();
	mMemory.release();
	mHashEntries.release();
	mVisibleEntries.release();
	mVoxelBlocks.release();
	mORBKeys.release();
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
	return map;
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
	return map;
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

void Mapping::IntegrateKeys(Frame& F) {

	std::vector<ORBKey> keys;
	cv::Mat desc;
	F.mDescriptors.download(desc);
	for (int i = 0; i < F.mNkp; ++i) {
		if (!F.mOutliers[i]) {
			ORBKey key;
			Eigen::Vector3d worldPos = F.Rotation() * F.mPoints[i] + F.Translation();
			key.pos = make_float3((float)worldPos(0), (float)worldPos(1), (float)worldPos(2));
			for (int j = 0; j < 32; ++j)
				key.descriptor[j] = desc.at<char>(i, j);
			keys.push_back(key);
		}
	}

	DeviceArray<ORBKey> dKeys(keys.size());
	dKeys.upload((void*) keys.data(), keys.size());

	InsertKeys(*this, dKeys);
}

void Mapping::GetORBKeys(DeviceArray<ORBKey>& keys, int& mnMapPoints) {
	CollectKeys(*this, keys, mnMapPoints);
}

void Mapping::GetKeysHost(std::vector<ORBKey>& vkeys) {

	int n;
	DeviceArray<ORBKey> keys;
	CollectKeys(*this, keys, n);

	if (n == 0)
		return;

	ORBKey* MapKeys = (ORBKey*) malloc(sizeof(ORBKey) * n);
	keys.download((void*) MapKeys, n);
	for (int i = 0; i < n; ++i) {
		ORBKey& key = MapKeys[i];
		vkeys.push_back(key);
	}
	delete [] MapKeys;
}

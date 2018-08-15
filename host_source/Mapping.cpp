#include "Mapping.hpp"

bool Mapping::mbFirstCall = true;

Mapping::Mapping() {}

Mapping::~Mapping() {
	ReleaseDeviceMemory();
}

void Mapping::AllocateDeviceMemory(MapDesc desc) {
	mMemory.create(NUM_SDF_BLOCKS);
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(NUM_BUCKETS);
	mHashEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVisibleEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVoxelBlocks.create(NUM_SDF_BLOCKS * BLOCK_SIZE);
}

void Mapping::ReleaseDeviceMemory() {
	mUsedMem.release();
	mNumVisibleEntries.release();
	mBucketMutex.release();
	mMemory.release();
	mHashEntries.release();
	mVisibleEntries.release();
	mVoxelBlocks.release();
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

#include "Map.h"

Map::Map() {}

Map::~Map() {
	ReleaseDeviceMemory();
}

void Map::AllocateDeviceMemory(MapDesc desc) {
//	mUsedMem.create(1);
//	mNumVisibleEntries.create(1);
//	mBucketMutex.create(desc.numBuckets);
//	mMemory.create(desc.numBuckets * desc.bucketSize);
//	mHashEntries.create(desc.numBuckets * desc.bucketSize);
//	mVisibleEntries.create(desc.numBuckets * desc.bucketSize);
//	mVoxelBlocks.create(desc.numBlocks * desc.blockSize3);
//	UpdateDesc(desc);
//
//	mMapPoints.create(desc.numBuckets);
//	mDescriptors.create(32, desc.numBuckets);
	mMemory.create(NUM_SDF_BLOCKS);
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(NUM_BUCKETS);
	mHashEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVisibleEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVoxelBlocks.create(NUM_SDF_BLOCKS * BLOCK_SIZE);
}

void Map::ReleaseDeviceMemory() {
	mUsedMem.release();
	mNumVisibleEntries.release();
	mBucketMutex.release();
	mMemory.release();
	mHashEntries.release();
	mVisibleEntries.release();
	mVoxelBlocks.release();
	mMapPoints.release();
	mDescriptors.release();
}

Map::operator DeviceMap() {
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

Map::operator const DeviceMap() const {
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

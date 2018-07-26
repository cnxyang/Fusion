#include "Map.h"

bool Map::mbFirstCall = true;

Map::Map() {}

Map::~Map() {
	ReleaseDeviceMemory();
}

void Map::AllocateDeviceMemory(MapDesc desc) {
	mMemory.create(NUM_SDF_BLOCKS);
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(NUM_BUCKETS);
	mHashEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVisibleEntries.create(NUM_BUCKETS    * BUCKET_SIZE);
	mVoxelBlocks.create(NUM_SDF_BLOCKS * BLOCK_SIZE);

	mDescriptors.create(MaxNoKeyPoints, 32, CV_8UC1);
}

void Map::ReleaseDeviceMemory() {
	mUsedMem.release();
	mNumVisibleEntries.release();
	mBucketMutex.release();
	mMemory.release();
	mHashEntries.release();
	mVisibleEntries.release();
	mVoxelBlocks.release();
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

void Map::SetFirstFrame(Frame& frame) {
	mMapPoints = frame.mMapPoints;
	frame.mDescriptors.copyTo(mDescriptors);
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

void Map::FuseKeyPoints(const Frame& frame) {
	if(mbFirstCall) {
		if(frame.mNkp > 0) {
//			AppendMapPoints(frame.mMapPoints, frame.mDescriptors, mMapPoints, mDescriptors, 0, frame.mNkp - 1);
//			GenerateIndexArray(frame.mMapPoints, mIndexArray, frame.mNkp);
		}
		mbFirstCall = false;
	}
}

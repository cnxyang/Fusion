#include "Map.h"

Map::Map() {}

Map::~Map() {
	ReleaseDeviceMemory();
}

void Map::AllocateDeviceMemory(MapDesc& desc) {
	mUsedMem.create(1);
	mNumVisibleEntries.create(1);
	mBucketMutex.create(desc.numBuckets);
	mMemory.create(desc.numBuckets * desc.bucketSize);
	mHashEntries.create(desc.numBuckets * desc.bucketSize);
	mVisibleEntries.create(desc.numBuckets * desc.bucketSize);
	mVoxelBlocks.create(desc.numBlocks * desc.blockSize3);
	UpdateDesc(desc);

	mMapPoints.create(desc.numBuckets);
	mDescriptors.create(32, desc.numBuckets);
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

}

Map::operator const DeviceMap() const {

}

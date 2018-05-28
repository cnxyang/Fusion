#ifndef __MAP_H__
#define __MAP_H__

#include "DeviceArray.h"
#include "DeviceFunc.h"
#include "DeviceMath.h"

struct VoxelMapDesc
{
	float voxelSize;
	float minDepthRange;
	float maxDepthRange;
	float truncateRatio;

	uint noBuckets;
	uint noBlocks;
	uint hashMask;
	uint bucketSize;
	uint blockSize;
	uint blockSize3;
	uint maxLinkedListSize;
	uint minMaxSubSample;
	uint renderingBlockSizeX;
	uint renderingBlockSizeY;
};

struct HashEntry {
	int3 pos;
	int ptr;
	int offset;
};

struct Voxel {
	short sdf;
	short sdfW;
	uchar3 rgb;
	short rgbW;
};

struct DeviceMap {
	PtrSz<uint> memory;
	PtrSz<int> usedMem;
	PtrSz<int> noVisibleEntries;
	PtrSz<int> bucketMutex;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;

	__device__ inline bool SearchVoxel(const float3&, Voxel&);
};

#endif

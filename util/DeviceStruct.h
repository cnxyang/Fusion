#ifndef __DEVICE_STRUCT_H__
#define __DEVICE_STRUCT_H__

#include "DeviceArray.h"

struct HashEntry {
	int3 pos;
	int ptr;
	int offset;
};

struct Point {
	float3 pos;
	int ptr;
	int id;
	bool valid;
};

struct Voxel {
	short sdf;
	short sdfW;
	uchar3 rgb;
	short rgbW;
};

struct MapDesc {
	int bucketSize;
	int numBuckets;
	int numBlocks;
	int hashMask;
	int blockSize;
	int blockSize3;
	int maxLinkedList;
	float voxelSize;
};

struct Rendering {
	int cols, rows;
	float fx, fy, cx, cy;
	float maxD, minD;
	DeviceArray2D<float4> VMap;
	DeviceArray2D<float3> NMap;
	DeviceArray2D<unsigned char> Render;
};

struct DeviceMap {
	PtrSz<int> memory;
	PtrSz<int> usedMem;
	PtrSz<int> bucketMutex;
	PtrSz<int> numVisibleEntries;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;
	PtrSz<Point> keyPoints;
	PtrStep<char> descriptors;
};

#endif


#ifndef __DEVICE_STRUCT_H__
#define __DEVICE_STRUCT_H__

#include "DeviceArray.h"

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

struct Corresp {
	int u, v;
	bool bICP, bRGB;
	float3 nlast, nvcross;
	float ICPres, RGBres;
};

struct DeviceMap {
	PtrSz<uint> memory;
	PtrSz<uint> uesdMem;
	PtrSz<uint> bucketMutex;
	PtrSz<uint> noVisibleEntries;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;
};

#endif

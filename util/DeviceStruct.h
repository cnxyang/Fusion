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

struct DeviceMap {
	PtrSz<uint> memory;
	PtrSz<uint> uesdMem;
	PtrSz<uint> bucketMutex;
	PtrSz<uint> noVisibleEntries;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;
};

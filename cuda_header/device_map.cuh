#ifndef DEVICE_MAP_HPP__
#define DEVICE_MAP_HPP__

#include "Mapping.hpp"
#include "device_array.hpp"
#include "device_math.hpp"

struct HBlock {
	int3 pos;
	int ptr;
	int next;
};

struct HVoxel {
	int w;
};

struct ORBKey {
	bool valid;
	float3 pos;
	float3 normal;
	uint nextKey;
	uint referenceKF;
	char descriptor[32];
};

class DMap {

	const int BlockSize = 8;
	const int BlockSize3 = 512;
	const int MaxBlocks = 0x30000;
	const int MaxVoxels = 0x50000;
	const int MaxBlockChain = 5;

public:
	__device__ int Hash(const int3& pos);
	__device__ int AllocateMem();
	__device__ void ReleaseMem(int idx);
	__device__ void ResetDeviceMem(int idx);
	__device__ HVoxel* FindVoxel(const int3& pos);
	__device__ HBlock* FindBlock(const int3& pos);

private:

	PtrSz<int> StackMem;
	PtrSz<int> StackPtr;
	PtrSz<HBlock> Blocks;
	PtrSz<HVoxel> Voxels;
};

class KeyMap {

	static constexpr float GridSize = 0.05;
	const int MaxKeys = 1000000;
	const int nBuckets = 5;

public:
	__device__ int Hash(const int3& pos);
	__device__ ORBKey* FindKey(const float3& pos);
	__device__ void InsertKey(ORBKey* key);

public:
	PtrSz<ORBKey> Keys;
};

#endif

#include "device_map.cuh"

#define UINT_MAX 0x7fffffff * 2U + 1

__device__ int DMap::Hash(const int3& pos) {

	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791));
	res %= DMap::MaxBlocks;
	if (res < 0)
		res += DMap::MaxBlocks;

	return res;
}

__device__ HVoxel* DMap::FindVoxel(const int3& pos) {
	HBlock* block = FindBlock(pos);
	if (block->ptr < 0)
		return nullptr;

	int idx = 0;
	idx += pos.z * DMap::BlockSize * DMap::BlockSize;
	idx += pos.y * DMap::BlockSize;
	idx += pos.x;
	HVoxel* voxel = &Voxels[block->ptr + idx];

	return voxel;
}

__device__ HBlock* DMap::FindBlock(const int3& pos) {

	int idx = Hash(pos);
	HBlock* block = &Blocks[idx];
	int counter = 0;

	while (block->ptr != -1) {
		if (block->pos == pos)
			return block;

		if (block->next == -1 ||
		    counter >= 5)
			return nullptr;

		counter++;
		block = &Blocks[block->next];
	}

	return nullptr;
}

__device__ int DMap::AllocateMem() {
	uint ptr = atomicAdd(StackPtr, 1);
	if (ptr >= 0 && ptr < DMap::MaxBlocks)
		return StackMem[ptr];
	else
		return -1;
}

__device__ void DMap::ReleaseMem(int idx) {
	uint ptr = atomicSub(StackPtr, 1);
	if (ptr >= 0 && ptr < DMap::MaxBlocks)
		StackMem[ptr] = idx;
}

__device__ void DMap::ResetDeviceMem(int idx) {
	if(idx >= 0 && idx < DMap::MaxVoxels) {
		if(idx == 0)
			StackPtr[0] = 0;
		Voxels[idx].w = 0;
		StackMem[idx] = idx;
	}

	if(idx >= 0 && idx < DMap::MaxBlocks) {
		Blocks[idx].ptr = -1;
		Blocks[idx].next = -1;
	}
}

__device__ ORBKey* KeyMap::FindKey(const float3& pos) {

	float3 gridPos = pos / GridSize;
	int idx = Hash(make_int3(gridPos.x, gridPos.y, gridPos.z));
	int bucketIdx = idx * nBuckets;
	for(int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		ORBKey* key = &Keys[bucketIdx];
		if(norm(key->pos - pos) <= 10) {
			return key;
		}
	}
	return nullptr;
}

__device__ void KeyMap::InsertKey(ORBKey* key) {

	ORBKey* oldKey = FindKey(key->pos);
	if(!oldKey) {

	}
	else {

	}
}

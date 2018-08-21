#include "device_map.cuh"

__device__ int KeyMap::Hash(const int3& pos) {

	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791));
	res %= KeyMap::MaxKeys;
	if (res < 0)
		res += KeyMap::MaxKeys;

	return res;
}

__device__ ORBKey* KeyMap::FindKey(const float3& pos) {

	float3 gridPos = pos / GridSize;
	int idx = Hash(make_int3(gridPos.x, gridPos.y, gridPos.z));
	int bucketIdx = idx * nBuckets;
	const float radius = 10.0f;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		ORBKey* key = &Keys[bucketIdx];
		if (key->valid && norm(key->pos - pos) <= radius) {
			return key;
		}
	}
	return nullptr;
}

__device__ ORBKey* KeyMap::FindKey(const float3& pos, int& first, int& buck) {

	first = -1;
	float3 gridPos = pos / GridSize;
	int3 p = make_int3((int)gridPos.x, (int)gridPos.y, (int)gridPos.z);
	int idx = Hash(p);

	int bucketIdx = buck = idx * nBuckets;
	const float radius = GridSize;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		ORBKey* key = &Keys[bucketIdx];
		if (!key->valid && first == -1)
			first = bucketIdx;

		if (key->valid && norm(key->pos - pos) <= radius) {
			return key;
		}
	}
	return nullptr;
}

__device__ void KeyMap::InsertKey(ORBKey* key) {

	int first = -1;
	int buck = 0;
	ORBKey* oldKey = FindKey(key->pos, first, buck);
	if (oldKey && oldKey->valid) {
		return;
	}
	else if (first != -1) {
		int lock = atomicExch(&Mutex[buck], 1);
		if(lock < 0) {
			ORBKey* oldkey = &Keys[first];
			memcpy((void*) oldkey, (void*) key, sizeof(ORBKey));
		}
		atomicExch(&Mutex[buck], -1);
		return;
	}
}

__device__ void KeyMap::ResetKeys(int index) {

	if (index < Mutex.size)
		Mutex[index] = -1;

	if (index < Keys.size) {
		Keys[index].valid = false;
		Keys[index].nextKey = -1;
		Keys[index].referenceKF = -1;
	}
}

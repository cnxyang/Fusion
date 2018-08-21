#ifndef DEVICE_MAP_HPP__
#define DEVICE_MAP_HPP__

#include "Mapping.hpp"
#include "device_array.hpp"
#include "device_math.hpp"


struct ORBKey {
	bool valid;
	float3 pos;
	uint nextKey;
	uint referenceKF;
	char descriptor[32];
};

struct KeyMap {

	static constexpr float GridSize = 0.03;
	static const int MaxKeys = 1000000;
	static const int nBuckets = 5;

public:
	__device__ int Hash(const int3& pos);
	__device__ ORBKey* FindKey(const float3& pos);
	__device__ ORBKey* FindKey(const float3& pos, int& first, int& buck);
	__device__ void InsertKey(ORBKey* key);
	__device__ void ResetKeys(int index);

public:
	PtrSz<ORBKey> Keys;
	PtrSz<int> Mutex;
};

#endif

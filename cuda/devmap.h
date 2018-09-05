#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include "mathlib.h"
#include "cuarray.h"

#define MaxThread 1024

struct RenderingBlock {

	int2 upperLeft;
	int2 lowerRight;
	float2 zRange;
};

struct HashEntry {

	int3 pos;
	int  ptr;
	int  offset;

	__device__ HashEntry();
	__device__ HashEntry(const HashEntry& other);
	__device__ HashEntry(int3 pos_, int ptr, int offset) ;
	__device__ void release();
	__device__ void operator=(const HashEntry& other);
	__device__ bool operator==(const int3& pos_) const;
	__device__ bool operator==(const HashEntry& other);
};

enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2 };

struct Voxel {

	short sdf;
	short sdfW;
	uchar3 rgb;
	short rgbW;
	static const int MaxWeight = 100;
	static const int MaxShort = 32767;

	__device__ Voxel();
	__device__ Voxel(float, short, uchar3, short);
	__device__ void release();
	__device__ float GetSdf() const;
	__device__ void SetSdf(float);
	__device__ void GetSdfAndColor(float & sdf, uchar3 & color) const;
	__device__ void operator+=(const Voxel& other);
	__device__ void operator-=(const Voxel& other);
	__device__ void operator=(const Voxel& other);
};

struct HostMap {

};

struct DeviceMap {

	__device__ uint Hash(const int3 & pos);
	__device__ Voxel FindVoxel(const int3& pos);
	__device__ Voxel FindVoxel(const float3& pos);
	__device__ HashEntry FindEntry(const int3& pos);
	__device__ HashEntry FindEntry(const float3& pos);
	__device__ void CreateBlock(const int3& blockPos);
	__device__ bool FindVoxel(const int3& pos, Voxel& vox);
	__device__ bool FindVoxel(const float3& pos, Voxel& vox);
	__device__ HashEntry CreateEntry(const int3& pos, const int& offset);

	__device__ int3 worldPosToVoxelPos(float3 pos) const;
	__device__ float3 worldPosToVoxelPosF(float3 pos) const;
	__device__ float3 voxelPosToWorldPos(int3 pos) const;
	__device__ int3 voxelPosToBlockPos(const int3& pos) const;
	__device__ int3 blockPosToVoxelPos(const int3& pos) const;
	__device__ int3 voxelPosToLocalPos(const int3& pos) const;
	__device__ int localPosToLocalIdx(const int3& pos) const;
	__device__ int3 localIdxToLocalPos(const int& idx) const;
	__device__ int3 worldPosToBlockPos(const float3& pos) const;
	__device__ float3 blockPosToWorldPos(const int3& pos) const;
	__device__ int voxelPosToLocalIdx(const int3& pos) const;

	static constexpr uint BlockSize = 8;
	static constexpr uint BlockSize3 = 512;
	static constexpr float DepthMin = 0.1f;
	static constexpr float DepthMax = 3.0f;
	static constexpr uint NumExcess = 500000;
	static constexpr uint NumBuckets = 1000000;
	static constexpr uint NumSdfBlocks = 400000;
	static constexpr uint MaxTriangles = 2000 * 2000;
	static constexpr float VoxelSize = 0.005f;
	static constexpr float TruncateDist = 0.03f;
	static constexpr int MaxRenderingBlocks = 260000;
	static constexpr float voxelSizeInv = 1.0 / VoxelSize;
	static constexpr float blockWidth = VoxelSize * BlockSize;
	static constexpr uint NumEntries = NumBuckets + NumExcess;

	PtrSz<int> heapMem;
	PtrSz<int> entryPtr;
	PtrSz<int> heapCounter;
	PtrSz<int> bucketMutex;
	PtrSz<Voxel> voxelBlocks;
	PtrSz<uint> noVisibleBlocks;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
};

struct ORBKey {
	bool valid;
	int obs;
	float3 pos;
	float3 normal;
	char descriptor[32];
};

struct KeyMap {

	static constexpr float GridSize = 0.03;
	static const int MaxKeys = 1000000;
	static const int nBuckets = 5;
	static const int MaxObs = 10;
	static const int MinObsThresh = -5;

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

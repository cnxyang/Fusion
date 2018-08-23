#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include <Eigen/Dense>
#include "device_array.hpp"
#include "device_math.hpp"

#define MaxThread 1024
#define DEV __device__
#define HOST __host__

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
	Matrix3f Rview, invRview;
	float3 tview;
	DeviceArray2D<float4> VMap;
	DeviceArray2D<float3> NMap;
	DeviceArray2D<uchar4> Render;
};

struct RenderingBlock {

	int2 upperLeft;
	int2 lowerRight;
	float2 zRange;
};

struct HashEntry {

	int3 pos;
	int  ptr;
	int  offset;

	DEV HashEntry();
	DEV HashEntry(const HashEntry& other);
	DEV HashEntry(int3 pos_, int ptr, int offset) ;
	DEV void release();
	DEV void operator=(const HashEntry& other);
	DEV bool operator==(const int3& pos_) const;
	DEV bool operator==(const HashEntry& other);
};

enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2 };

struct Voxel {

	short sdf;
	short sdfW;
	static const uint MAX_WEIGHT = 100;
	static const int MaxShort = 32767;

	DEV Voxel();
	DEV Voxel(float sdf, short weight);
	DEV void release();
	DEV float GetSdf() const;
	DEV void SetSdf(float);
	DEV void operator+=(const Voxel& other);
	DEV void operator-=(const Voxel& other);
	DEV void operator=(const Voxel& other);
};

struct HostMap {

};

struct DeviceMap {

	HOST void Release();
	HOST void Download(DeviceMap& map);

	DEV uint Hash(const int3 & pos);
	DEV Voxel FindVoxel(const int3& pos);
	DEV Voxel FindVoxel(const float3& pos);
	DEV HashEntry FindEntry(const int3& pos);
	DEV HashEntry FindEntry(const float3& pos);
	DEV void CreateBlock(const int3& blockPos);
	DEV bool FindVoxel(const int3& pos, Voxel& vox);
	DEV bool FindVoxel(const float3& pos, Voxel& vox);
	DEV HashEntry CreateEntry(const int3& pos, const int& offset);

	DEV int3 worldPosToVoxelPos(float3 pos) const;
	DEV float3 worldPosToVoxelPosF(float3 pos) const;
	DEV float3 voxelPosToWorldPos(int3 pos) const;
	DEV int3 voxelPosToBlockPos(const int3& pos) const;
	DEV int3 blockPosToVoxelPos(const int3& pos) const;
	DEV int3 voxelPosToLocalPos(const int3& pos) const;
	DEV int localPosToLocalIdx(const int3& pos) const;
	DEV int3 localIdxToLocalPos(const int& idx) const;
	DEV int3 worldPosToBlockPos(const float3& pos) const;
	DEV float3 blockPosToWorldPos(const int3& pos) const;
	DEV int voxelPosToLocalIdx(const int3& pos) const;

	static constexpr uint BlockSize = 8;
	static constexpr uint BlockSize3 = 512;
	static constexpr float DepthMin = 0.1f;
	static constexpr float DepthMax = 3.5f;
	static constexpr uint NumExcess = 500000;
	static constexpr uint NumBuckets = 1000000;
	static constexpr uint NumSdfBlocks = 750000;
	static constexpr float VoxelSize = 0.005f;
	static constexpr float TruncateDist = 0.03f;
	static constexpr int MaxRenderingBlocks = 260000;
	static constexpr uint NumEntries = NumBuckets + NumExcess;

	PtrSz<int> heapMem;
	PtrSz<int> entryPtr;
	PtrSz<int> heapCounter;
	PtrSz<int> bucketMutex;
	PtrSz<Voxel> voxelBlocks;
	PtrSz<uint> noVisibleBlocks;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;

	PtrSz<Voxel> SdfBlocks;
	PtrSz<uint> BucketMutex;
	PtrSz<uint> BlockMemPtr;
	PtrSz<uint> EntryMemPtr;
	PtrSz<uint> BlockMemStack;
	PtrSz<uint> EntryMemStack;
	PtrSz<uint> EntryIdSwapout;
	PtrSz<uint> EntryIdVisible;
	PtrSz<HashEntry> HashTable;
	PtrSz<HashEntry> HashTableTemp;
};

struct ORBKey {
	bool valid;
	float3 pos;
	float3 normal;
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

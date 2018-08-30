#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include <Eigen/Dense>
#include "device_array.hpp"
#include "device_math.hpp"

#define MaxThread 1024

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

	DEV_FUNC HashEntry();
	DEV_FUNC HashEntry(const HashEntry& other);
	DEV_FUNC HashEntry(int3 pos_, int ptr, int offset) ;
	DEV_FUNC void release();
	DEV_FUNC void operator=(const HashEntry& other);
	DEV_FUNC bool operator==(const int3& pos_) const;
	DEV_FUNC bool operator==(const HashEntry& other);
};

enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2 };

struct Voxel {

	short sdf;
	short sdfW;
	uchar3 rgb;
	short rgbW;
	static const int MaxWeight = 100;
	static const int MaxShort = 32767;

	DEV_FUNC Voxel();
	DEV_FUNC Voxel(float, short, uchar3, short);
	DEV_FUNC void release();
	DEV_FUNC float GetSdf() const;
	DEV_FUNC void SetSdf(float);
	DEV_FUNC void operator+=(const Voxel& other);
	DEV_FUNC void operator-=(const Voxel& other);
	DEV_FUNC void operator=(const Voxel& other);
};

struct HostMap {

};

struct DeviceMap {

	HOST_FUNC void Release();
	HOST_FUNC void Download(DeviceMap& map);

	DEV_FUNC uint Hash(const int3 & pos);
	DEV_FUNC Voxel FindVoxel(const int3& pos);
	DEV_FUNC Voxel FindVoxel(const float3& pos);
	DEV_FUNC HashEntry FindEntry(const int3& pos);
	DEV_FUNC HashEntry FindEntry(const float3& pos);
	DEV_FUNC void CreateBlock(const int3& blockPos);
	DEV_FUNC bool FindVoxel(const int3& pos, Voxel& vox);
	DEV_FUNC bool FindVoxel(const float3& pos, Voxel& vox);
	DEV_FUNC HashEntry CreateEntry(const int3& pos, const int& offset);

	DEV_FUNC int3 worldPosToVoxelPos(float3 pos) const;
	DEV_FUNC float3 worldPosToVoxelPosF(float3 pos) const;
	DEV_FUNC float3 voxelPosToWorldPos(int3 pos) const;
	DEV_FUNC int3 voxelPosToBlockPos(const int3& pos) const;
	DEV_FUNC int3 blockPosToVoxelPos(const int3& pos) const;
	DEV_FUNC int3 voxelPosToLocalPos(const int3& pos) const;
	DEV_FUNC int localPosToLocalIdx(const int3& pos) const;
	DEV_FUNC int3 localIdxToLocalPos(const int& idx) const;
	DEV_FUNC int3 worldPosToBlockPos(const float3& pos) const;
	DEV_FUNC float3 blockPosToWorldPos(const int3& pos) const;
	DEV_FUNC int voxelPosToLocalIdx(const int3& pos) const;

	static constexpr uint BlockSize = 8;
	static constexpr uint BlockSize3 = 512;
	static constexpr float DepthMin = 0.1f;
	static constexpr float DepthMax = 2.0f;
	static constexpr uint NumExcess = 500000;
	static constexpr uint NumBuckets = 1000000;
	static constexpr uint NumSdfBlocks = 400000;
	static constexpr uint MaxTriangles = 2000 * 2000;
	static constexpr float VoxelSize = 0.006f;
	static constexpr float TruncateDist = 0.035f;
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

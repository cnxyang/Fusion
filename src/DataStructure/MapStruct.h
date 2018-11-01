#pragma once

#include "VectorMath.h"
#include "DeviceArray.h"

#define MaxThread 1024

enum {
	EntryAvailable = -1,
	EntryOccupied = -2
};

struct MapState
{
	// constants shouldn't be changed at all
	// these are down to the basic design of the system
	// changes made to these will render system unstable
	// if not unusable at all.
	int blockSize;
	int blockSize3;
	float blockWidth;

	// parameters control how far the camera sees
	// should keep them in minimum as long as they
	// satisfy your needs. Larger viewing frusta
	// will significantly slow down the system.
	// as more sdf blocks will be allocated.
	float depthMin_raycast;
	float depthMax_raycast;
	float depthMin_preprocess;
	float depthMax_preprocess;

	// parameters that control the size of the
	// device memory needed in the allocation stage.
	// Note that numBuckets should always be bigger
	// than numSdfBlock as that's the requirement
	// of hash table;
	int maxNumBuckets;
	int maxNumVoxelBlocks;
	int maxNumMeshTriangles;
	int maxNumMeshVertices;
	int maxNumHashEntries;
	int maxNumRenderingBlocks;

	// parameters that won't affect system performance
	// too much, generally just affect the appearance
	// of the map and are free to be modified.
	// Note that due to imperfections in the function
	// PARRALLEL SCAN, too large voxelSize will not work.
	float voxelSize;
	float invVoxelSize;
	float truncateDistance;
	float stepScale_raycast;
};

__device__ extern MapState mapState;

inline void updateMapState(MapState state)
{
	SafeCall(cudaMemcpyToSymbol(&mapState, &state, sizeof(MapState)));
}

inline void downloadMapState(MapState& state)
{
	SafeCall(cudaMemcpyFromSymbol(&state, &mapState, sizeof(MapState)));
}

struct __align__(8) RenderingBlock
{
	short2 upperLeft;
	short2 lowerRight;
	float2 zRange;
};

struct __align__(8) Voxel
{
	float sdf;
	unsigned char weight;
	uchar3 color;

	__device__ __forceinline__ Voxel();
	__device__ __forceinline__ Voxel(float sdf, short weight, uchar3 rgb);
	__device__ __forceinline__ void release();
	__device__ __forceinline__ void getValue(float& sdf, uchar3& rgb) const;
	__device__ __forceinline__ void operator=(const Voxel& other);
};

struct __align__(16) HashEntry
{
	int ptr;
	int offset;
	int3 pos;

	__device__ __forceinline__ HashEntry();
	__device__ __forceinline__ HashEntry(int3 pos, int ptr, int offset);
	__device__ __forceinline__ HashEntry(const HashEntry& other);
	__device__ __forceinline__ void release();
	__device__ __forceinline__ void operator=(const HashEntry& other);
	__device__ __forceinline__ bool operator==(const int3& pos) const;
	__device__ __forceinline__ bool operator==(const HashEntry& other) const;
};

struct MapStruct
{
	__device__ uint Hash(const int3 & pos);
	__device__ Voxel FindVoxel(const int3 & pos);
	__device__ Voxel FindVoxel(const float3 & pos);
	__device__ Voxel FindVoxel(const float3 & pos, HashEntry & cache, bool & valid);
	__device__ HashEntry FindEntry(const int3 & pos);
	__device__ HashEntry FindEntry(const float3 & pos);
	__device__ void CreateBlock(const int3 & blockPos);
	__device__ bool FindVoxel(const int3 & pos, Voxel & vox);
	__device__ bool FindVoxel(const float3 & pos, Voxel & vox);
	__device__ HashEntry CreateEntry(const int3 & pos, const int & offset);

	__device__ int3 worldPosToVoxelPos(float3 pos) const;
	__device__ int3 voxelPosToBlockPos(const int3 & pos) const;
	__device__ int3 blockPosToVoxelPos(const int3 & pos) const;
	__device__ int3 voxelPosToLocalPos(const int3 & pos) const;
	__device__ int3 localIdxToLocalPos(const int & idx) const;
	__device__ int3 worldPosToBlockPos(const float3 & pos) const;
	__device__ float3 worldPosToVoxelPosF(float3 pos) const;
	__device__ float3 voxelPosToWorldPos(int3 pos) const;
	__device__ float3 blockPosToWorldPos(const int3 & pos) const;
	__device__ int localPosToLocalIdx(const int3 & pos) const;
	__device__ int voxelPosToLocalIdx(const int3 & pos) const;

	static constexpr uint BlockSize = 8;
	static constexpr uint BlockSize3 = 512;
	static constexpr float DepthMin = 0.1f;
	static constexpr float DepthMax = 3.0f;
	static constexpr uint NumExcess = 500000;
	static constexpr uint NumBuckets = 1000000;
	static constexpr uint NumSdfBlocks = 700000;
	static constexpr uint NumVoxels = NumSdfBlocks * BlockSize3;
	static constexpr uint MaxTriangles = 20000000; // roughly 700MB memory
	static constexpr uint MaxVertices = MaxTriangles * 3;
	static constexpr float VoxelSize = 0.006f;
	static constexpr float TruncateDist = VoxelSize * 8;
	static constexpr int MaxRenderingBlocks = 260000;
	static constexpr float voxelSizeInv = 1.0 / VoxelSize;
	static constexpr float blockWidth = VoxelSize * BlockSize;
	static constexpr uint NumEntries = NumBuckets + NumExcess;
	static constexpr float stepScale = 0.5 * TruncateDist * voxelSizeInv;

	PtrSz<int> heapMem;
	PtrSz<int> entryPtr;
	PtrSz<int> heapCounter;
	PtrSz<int> bucketMutex;
	PtrSz<Voxel> voxelBlocks;
	PtrSz<uint> noVisibleBlocks;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;

	__host__ void allocateHostMemory();
	__host__ void allocateDeviceMemory();
	__host__ void releaseHostMemory();
	__host__ void releaseDeviceMemory();
};

__device__ __forceinline__ HashEntry::HashEntry() :
	pos(make_int3(0)), ptr(-1), offset(-1)
{
}

__device__ __forceinline__ HashEntry::HashEntry(int3 pos, int ptr, int offset) :
	pos(pos), ptr(ptr), offset(offset)
{
}

__device__ __forceinline__ HashEntry::HashEntry(const HashEntry& other)
{
	pos = other.pos;
	ptr = other.ptr;
	offset = other.offset;
}

__device__ __forceinline__ void HashEntry::release()
{
	ptr = -1;
}

__device__ __forceinline__ void HashEntry::operator=(const HashEntry& other)
{
	pos = other.pos;
	ptr = other.ptr;
	offset = other.offset;
}

__device__ __forceinline__ bool HashEntry::operator==(const int3& pos) const
{
	return (this->pos == pos);
}

__device__ __forceinline__ bool HashEntry::operator==(const HashEntry& other) const
{
	return other.pos == pos;
}

__device__ __forceinline__ Voxel::Voxel()
: sdf(std::nanf("0x7fffffff")), weight(0), color(make_uchar3(0))
{
}

__device__ __forceinline__ Voxel::Voxel(float sdf, short weight, uchar3 rgb)
: sdf(sdf), weight(weight), color(rgb)
{
}

__device__ __forceinline__ void Voxel::release()
{
	sdf = std::nanf("0x7fffffff");
	weight = 0;
	color = make_uchar3(0);
}

__device__ __forceinline__ void Voxel::getValue(float& sdf, uchar3& color) const
{
	sdf = this->sdf;
	color = this->color;
}

__device__ __forceinline__ void Voxel::operator=(const Voxel& other)
{
	sdf = other.sdf;
	weight = other.weight;
	color = other.color;
}

struct SURF
{
	bool valid;
	float3 pos;
	float4 normal;
	float descriptor[64];
};

struct KeyMap {

	static constexpr float GridSize = 0.01;
	static constexpr int MaxKeys = 100000;
	static constexpr int nBuckets = 5;
	static constexpr int maxEntries = MaxKeys * nBuckets;
	static constexpr int MaxObs = 10;
	static constexpr int MinObsThresh = -5;

	__device__ int Hash(const int3& pos);
	__device__ SURF * FindKey(const float3 & pos);
	__device__ SURF * FindKey(const float3 & pos, int & first, int & buck, int & hashIndex);
	__device__ void InsertKey(SURF* key, int & hashIndex);
	__device__ void ResetKeys(int index);

	PtrSz<SURF> Keys;
	PtrSz<int> Mutex;
};

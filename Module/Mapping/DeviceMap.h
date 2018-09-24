#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include "VectorMath.h"
#include "DeviceArray.h"

#define MaxThread 1024

enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2 };

struct __align__(8) RenderingBlock {

	short2 upperLeft;

	short2 lowerRight;

	float2 zRange;
};

struct __align__(16) HashEntry {

	__device__ __forceinline__ HashEntry() :
			pos(make_int3(0)), ptr(-1), offset(-1) {
	}

	__device__ __forceinline__ HashEntry(int3 pos_, int ptr_, int offset_) :
			pos(pos_), ptr(ptr_), offset(offset_) {
	}

	__device__ __forceinline__ HashEntry(const HashEntry & other) {
		pos = other.pos;
		ptr = other.ptr;
		offset = other.offset;
	}

	__device__ __forceinline__ void release() {
		ptr = -1;
	}

	__device__ __forceinline__ void operator=(const HashEntry & other) {
		pos = other.pos;
		ptr = other.ptr;
		offset = other.offset;
	}

	__device__ __forceinline__ bool operator==(const int3 & pos_) const {
		return pos == pos_;
	}

	__device__ __forceinline__ bool operator==(const HashEntry & other) const {
		return other.pos == pos;
	}

	int3 pos;

	int  ptr;

	int  offset;
};

struct __align__(8) Voxel {

	__device__ __forceinline__ Voxel() :
			sdf(std::nanf("0x7fffffff")), weight(0), color(make_uchar3(0)) {
	}

	__device__ __forceinline__ Voxel(float sdf_, short weight_, uchar3 color_) :
			sdf(sdf_), weight(weight_), color(color_) {
	}

	__device__ __forceinline__ void release() {
		sdf = std::nanf("0x7fffffff");
		weight = 0;
		color = make_uchar3(0);
	}

	__device__ __forceinline__ void getValue(float & sdf_, uchar3 & color_) const {
		sdf_ = sdf;
		color_ = color;
	}

	__device__ __forceinline__ void operator=(const Voxel & other) {
		sdf = other.sdf;
		weight = other.weight;
		color = other.color;
	}

	float sdf;

	unsigned char weight;

	uchar3 color;
};

struct KeyPoint {

};

struct SurfKey : public KeyPoint {

	int valid;

	float3 pos;

	float4 normal;

	float descriptor[64];
};

struct DeviceMap {

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
	static constexpr float VoxelSize = 0.008f;
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
};

struct KeyMap {

	static constexpr float GridSize = 0.03;
	static constexpr int MaxKeys = 100000;
	static constexpr int nBuckets = 5;
	static constexpr int maxEntries = MaxKeys * nBuckets;
	static constexpr int MaxObs = 10;
	static constexpr int MinObsThresh = -5;

public:
	__device__ int Hash(const int3& pos);
	__device__ SurfKey * FindKey(const float3 & pos);
	__device__ SurfKey * FindKey(const float3 & pos, int & first, int & buck, int & hashIndex);
	__device__ void InsertKey(SurfKey* key);
	__device__ void ResetKeys(int index);

public:
	PtrSz<SurfKey> Keys;
	PtrSz<int> Mutex;
};

#endif

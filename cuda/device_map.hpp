#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include <Eigen/Dense>
#include "device_array.hpp"
#include "device_math.hpp"

#define MaxThread 1024
#define DEV __device__
enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2 };

static const uint  NUM_BUCKETS 		 = 500000;
static const uint  NUM_SDF_BLOCKS 	 = 300000;
static const uint  BLOCK_DIM	 	 = 8;
static const uint  BUCKET_SIZE 		 = 5;
static const uint  LINKED_LIST_SIZE  = 7;
static const float VOXEL_SIZE 		 = 0.005f;
static const uint  BLOCK_SIZE		 = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
static const uint  NUM_ENTRIES		 = BUCKET_SIZE * NUM_BUCKETS;
static const float DEPTH_MIN		 = 0.1f;
static const float DEPTH_MAX		 = 3.0f;
static const float TRUNC_DIST		 = 0.03f;
static const int NUM_MAX_TRIANGLES = 2000 * 2000;
static const int MAX_RENDERING_BLOCKS = 65535 * 4;

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

struct Voxel {

	float sdf;
	uchar3 rgb;
	float sdfW;
	static const uint MAX_WEIGHT = 100;

	DEV Voxel();
	DEV void release();
	DEV void operator+=(const Voxel& other);
	DEV void operator-=(const Voxel& other);
	DEV void operator=(const Voxel& other);
};

struct DeviceMap {

	PtrSz<int> heapMem;
	PtrSz<int> heapCounter;
	PtrSz<uint> noVisibleBlocks;
	PtrSz<int> bucketMutex;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;

	DEV uint Hash(const int3 & pos);
	DEV HashEntry createHashEntry(const int3& pos, const int& offset);
	DEV void CreateBlock(const int3& blockPos);
	DEV bool searchVoxel(const float3& pos, Voxel& vox);
	DEV bool searchVoxel(const int3& pos, Voxel& vox);
	DEV Voxel searchVoxel(const int3& pos);
	DEV Voxel searchVoxel(const float3& pos);
	DEV HashEntry searchHashEntry(const float3& pos);
	DEV HashEntry searchHashEntry(const int3& pos);
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
};

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

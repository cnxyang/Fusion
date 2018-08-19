#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include <Eigen/Dense>
#include "device_array.hpp"
#include "device_math.hpp"

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

struct HashEntry
{
	int3 pos;
	int  ptr;
	int  offset;

#ifdef __CUDACC__

	__device__
	HashEntry() : pos(make_int3(0x7fffffff)), ptr(EntryAvailable), offset(0) {}

	__device__
	HashEntry(const HashEntry & other) { (*this)=(other); }

	__device__
	HashEntry(int3 pos_, int ptr, int offset) : pos(pos_), ptr(ptr), offset(offset) {}

	__device__
	void release()
	{
		pos = make_int3(0);
		offset = 0;
		ptr = EntryAvailable;
	}

	__device__
	void operator=(const HashEntry & other)
	{
		pos = other.pos;
		ptr = other.ptr;
		offset = other.offset;
	}

	__device__
	bool operator==(const int3 & pos_) const
	{
		return pos == pos_ && ptr != EntryAvailable;
	}

	__device__
	bool operator==(const HashEntry & other)
	{
		return (*this)==(other.pos);
	}
#endif
};

struct Voxel
{
	float sdf;
	uchar3 rgb;
	float sdfW;

	static const uint MAX_WEIGHT = 100;

#ifdef __CUDACC__

	__device__
	Voxel(): sdf((float)0x7fffffff), rgb(make_uchar3(0, 0, 0)), sdfW(0) {}

	__device__
	void release()
	{
		sdf = (float)0x7fffffff;
		sdfW = 0;
		rgb = make_uchar3(0, 0, 0);
	}

	__device__
	void operator+=(const Voxel & other)
	{

		sdf = (sdf * sdfW + other.sdf * other.sdfW ) / (sdfW + other.sdfW);
		sdfW += other.sdfW;
	}

	__device__
	void operator-=(const Voxel & other)
	{
		// TODO: de-fusion method
	}

	__device__
	void operator=(const Voxel & other)
	{
		sdf = other.sdf;
		rgb = other.rgb;
		sdfW = other.sdfW;
	}
#endif
};

#ifdef __CUDACC__
__device__ inline
uint computeHashPos(const int3 & pos, const int numBuckets)
{
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % numBuckets;

	if(res < 0) res += NUM_BUCKETS;
	return res;
}
#endif

struct DeviceMap {
	PtrSz<int> heapMem;
	PtrSz<int> heapCounter;
	PtrSz<int> noVisibleBlocks;
	PtrSz<int> bucketMutex;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	PtrSz<Voxel> voxelBlocks;
#ifdef __CUDACC__

	__device__
	bool searchVoxel(const float3 & pos, Voxel & vox)
	{
		int3 voxel_pos = worldPosToVoxelPos(pos);

		return searchVoxel(voxel_pos, vox);
	}

	__device__
	bool searchVoxel(const int3 & pos, Voxel & vox)
	{
		HashEntry entry = searchHashEntry(voxelPosToBlockPos(pos));
		if(entry.ptr == EntryAvailable) return false;

		int idx = voxelPosToLocalIdx(pos);

		vox = voxelBlocks[entry.ptr + idx];
		return true;
	}

	__device__
	Voxel searchVoxel(const int3 & pos)
	{
		HashEntry entry = searchHashEntry(voxelPosToBlockPos(pos));

		Voxel voxel;
		if(entry.ptr == EntryAvailable) return voxel;

		return voxelBlocks[entry.ptr + voxelPosToLocalIdx(pos)];
	}


	__device__
	Voxel searchVoxel(const float3 & pos)
	{
		int3 p = make_int3(pos);
		HashEntry entry = searchHashEntry(voxelPosToBlockPos(p));

		Voxel voxel;
		if(entry.ptr == EntryAvailable) return voxel;

		return voxelBlocks[entry.ptr + voxelPosToLocalIdx(p)];
	}

	/* ------------------------ *
	 * hash entry manipulations *
	 *------------------------- */

	__device__
	HashEntry searchHashEntry(const float3 & pos)
	{
		int3 blockIdx = worldPosToBlockPos(pos);

		return searchHashEntry(blockIdx);
	}

	__device__
	HashEntry searchHashEntry(const int3 & pos)
	{
		uint bucketIdx = computeHashPos(pos, NUM_BUCKETS);
		uint entryIdx = bucketIdx * BUCKET_SIZE;

		HashEntry entry(pos, EntryAvailable, 0);

		for(uint i = 0; i < BUCKET_SIZE; ++i, ++entryIdx)
		{
			HashEntry & curr = hashEntries[entryIdx];

			if(curr == entry) return curr;
		}

#ifdef HANDLE_COLLISION
		const uint lastEntryIdx = (bucketIdx + 1) * BUCKET_SIZE - 1;
		entryIdx = lastEntryIdx;

		for(int i = 0; i < LINKED_LIST_SIZE; ++i)
		{
			HashEntry & curr = hashEntries[entryIdx];

			if(curr == entry) return curr;

			if(curr.offset == 0) break;

			entryIdx = lastEntryIdx + curr.offset % (BUCKET_SIZE * NUM_BUCKETS);
		}
#endif

		return entry;
	}

	/* -------------------------------------- *
	 * really confusing coordinate converters *
	 *--------------------------------------- */

	__device__
	int3 worldPosToVoxelPos(float3 pos) const
	{
		float3 p = pos / VOXEL_SIZE;
		return make_int3(p);
	}

	__device__
	float3 worldPosToVoxelPosF(float3 pos) const
	{
		return pos / VOXEL_SIZE;
	}

	__device__
	float3 voxelPosToWorldPos(int3 pos) const
	{
		return pos * VOXEL_SIZE;
	}

	__device__
	int3 voxelPosToBlockPos(const int3 & pos) const
	{
		int3 voxel = pos;

		if (voxel.x < 0) voxel.x -= BLOCK_DIM - 1;
		if (voxel.y < 0) voxel.y -= BLOCK_DIM - 1;
		if (voxel.z < 0) voxel.z -= BLOCK_DIM - 1;

		return voxel / BLOCK_DIM;
	}

	__device__
	int3 blockPosToVoxelPos(const int3 & pos) const
	{
		return pos * BLOCK_DIM;
	}

	__device__
	int3 voxelPosToLocalPos(const int3 & pos) const
	{
		int3 local = pos % BLOCK_DIM;

		if(local.x < 0) local.x += BLOCK_DIM;
		if(local.y < 0) local.y += BLOCK_DIM;
		if(local.z < 0) local.z += BLOCK_DIM;

		return local;
	}

	__device__
	int localPosToLocalIdx(const int3 & pos) const
	{
		return pos.z * BLOCK_DIM * BLOCK_DIM + pos.y * BLOCK_DIM + pos.x;
	}

	__device__
	int3 localIdxToLocalPos(const int & idx) const
	{
		uint x = idx % BLOCK_DIM;
		uint y = idx % (BLOCK_DIM * BLOCK_DIM) / BLOCK_DIM;
		uint z = idx / (BLOCK_DIM * BLOCK_DIM);

		return make_int3(x, y, z);
	}

	__device__
	int3 worldPosToBlockPos(const float3 & pos) const
	{
		return voxelPosToBlockPos(worldPosToVoxelPos(pos));
	}

	__device__
	float3 blockPosToWorldPos(const int3 & pos) const
	{
		return voxelPosToWorldPos(blockPosToVoxelPos(pos));
	}

	__device__ __forceinline__
	int voxelPosToLocalIdx(const int3 & pos) const
	{
		return localPosToLocalIdx(voxelPosToLocalPos(pos));
	}

#endif
};

#endif

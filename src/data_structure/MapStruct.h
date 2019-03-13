#pragma once

#include "VectorMath.h"
#include "DeviceArray.h"

#define BLOCK_SIZE 8
#define BLOCK_SIZE_SUB_1 7
#define BLOCK_SIZE3 512
#define MAX_THREAD 1024

enum
{
	EntryAvailable = -1,
	EntryOccupied = -2
};

struct MapState
{
	int maxNumBuckets;
	int maxNumVoxelBlocks;
	int maxNumMeshTriangles;
	int maxNumHashEntries;
	int maxNumRenderingBlocks;

	float zmin_raycast_;
	float zmax_raycast_;
	float zmin_update_;
	float zmax_update_;
	float voxel_size_; // voxel size in meters.

	__device__ __host__ int num_total_voxel() const;
	__device__ __host__ int num_excess_entry() const;
	__device__ __host__ int num_total_mesh_vertices() const;
	__device__ __host__ float block_size_metric() const;
	__device__ __host__ float inverse_voxel_size() const;
	__device__ __host__ float truncation_dist() const;
	__device__ __host__ float raycast_step_scale() const;
};

struct __align__(8) RenderingBlock
{
	short2 upperLeft;
	short2 lowerRight;
	float2 zRange;
};

struct Voxel
{
	float sdf_;
	short weight_;
	uchar3 rgb_;

#ifdef __NVCC__
	__device__ Voxel();
	__device__ Voxel(float sdf, short weight, uchar3 rgb);
	__device__ void release();
	__device__ void getValue(float& sdf, uchar3& rgb) const;
	__device__ void operator=(const Voxel& other);
#endif
};

struct __align__(16) HashEntry
{
	int next;
	int offset;
	int3 pos;

#ifdef __NVCC__
	__device__ HashEntry();
	__device__ HashEntry(int3 pos, int next, int offset);
	__device__ HashEntry(const HashEntry& other);
	__device__ void release();
	__device__ void operator=(const HashEntry& other);
	__device__ bool operator==(const int3& pos) const;
	__device__ bool operator==(const HashEntry& other) const;
#endif
};

struct MapStruct
{
#ifdef __NVCC__
	__device__ uint Hash(const int3& pos) const;
	__device__ Voxel FindVoxel(const int3& pos);
	__device__ Voxel FindVoxel(const float3& pos);
	__device__ Voxel FindVoxel(const float3& pos, HashEntry& cache, bool& valid);
	__device__ HashEntry FindEntry(const int3& pos);
	__device__ HashEntry FindEntry(const float3& pos);
	__device__ void CreateBlock(const int3& blockPos);
	__device__ bool FindVoxel(const int3& pos, Voxel& vox);
	__device__ bool FindVoxel(const float3& pos, Voxel& vox);
	__device__ HashEntry create_entry(const int3& pos, const int& offset);

	__device__ void find_voxel(const int3 &voxel_pos, Voxel *&out) const;
	__device__ void find_entry(const int3 &block_pos, HashEntry *&out) const;

	__device__ int3 posWorldToVoxel(float3 pos) const;
	__device__ int3 posVoxelToBlock(const int3& pos) const;
	__device__ int3 posBlockToVoxel(const int3& pos) const;
	__device__ int3 posVoxelToLocal(const int3& pos) const;
	__device__ int3 posIdxToLocal(const int& idx) const;
	__device__ int3 posWorldToBlock(const float3& pos) const;
	__device__ int posLocalToIdx(const int3& pos) const;
	__device__ int posVoxelToIdx(const int3& pos) const;
	__device__ float3 posWorldToVoxelFloat(float3 pos) const;
	__device__ float3 posVoxelToWorld(int3 pos) const;
	__device__ float3 posBlockToWorld(const int3& pos) const;
	#endif

	int* heapMem;
	int* entryPtr;
	int* heapCounter;
	int* bucketMutex;
	Voxel* voxelBlocks;
	uint* noVisibleBlocks;
	HashEntry* hashEntries;
	HashEntry* visibleEntries;
};

extern bool state_initialised;
extern MapState state;
__device__ extern MapState param;

void update_device_map_state();

//=================================================
// TO BE DEPRECATED BELOW
//=================================================

struct SURF
{
	bool valid;
	float3 pos;
	float4 normal;
	float descriptor[64];
};

struct KeyMap
{

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

#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "System.h"
#include "Tracking.h"
#include "KeyFrame.h"
#include "DeviceMap.h"

#include <vector>
#include <opencv.hpp>

class KeyMap;
class System;
class Tracker;
class PointCloud;

class DistanceField
{
public:

	DistanceField();


	void Reset();

	void Release();

	void CreateModel();

	void UpdateMapKeys();

	void CreateRAM();

	void DownloadToRAM();

	void UploadFromRAM();

	void ReleaseRAM();

	bool HasNewKF();

	void FuseKeyFrame(const KeyFrame * kf);

	void FuseKeyPoints(Frame * f);

	void UpdateVisibility(const KeyFrame * kf, uint & no);

	void UpdateVisibility(Frame * f, uint & no);

	void DefuseColor(Frame * f, uint & no);

	void FuseColor(Frame * f, uint & no);

	void RayTrace(uint noVisibleBlocks, Frame * f);

	void RayTrace(uint noVisibleBlocks, KeyFrame * f);

	void RayTraceWithColor(uint noVisibleBlocks, Frame * f);

	void ForwardWarp(Frame * last, Frame * next);

	void UpdateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
			float depthMin, float depthMax, float fx, float fy, float cx,
			float cy, uint & no);

	void FuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color,
			const DeviceArray2D<float4> & normal, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void DefuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color,
			const DeviceArray2D<float4> & normal, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float4> & nmap, float depthMin, float depthMax,
			float fx, float fy, float cx, float cy);

	void RayTraceWithColor(uint noVisibleBlocks, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float4> & nmap, DeviceArray2D<uchar3> & color,
			float depthMin, float depthMax, float fx, float fy, float cx,
			float cy);

	operator KeyMap() const;

	operator DeviceMap() const;

	std::vector<KeyFrame *> LocalMap() const;
	std::vector<KeyFrame *> GlobalMap() const;

	std::atomic<bool> meshUpdated;
	std::atomic<bool> mapPointsUpdated;
	std::atomic<bool> mapUpdated;
	std::atomic<bool> hasNewKFFlag;
	bool lost;

	uint noKeysHost;
	uint noTrianglesHost;
	uint noBlocksInFrustum;
	DeviceArray<float3> modelVertex;
	DeviceArray<float3> modelNormal;
	DeviceArray<uchar3> modelColor;
	std::vector<SURF> hostKeys;

	std::vector<const KeyFrame *> localMap;
	std::set<const KeyFrame *> keyFrames;

	// Host Memory Spaces
	int * heapRAM;
	int * heapCounterRAM;
	int * hashCounterRAM;
	int * bucketMutexRAM;
	Voxel * sdfBlockRAM;
	uint * noVisibleEntriesRAM;
	HashEntry * hashEntriesRAM;
	HashEntry * visibleEntriesRAM;

	int * mutexKeysRAM;
	SURF * mapKeysRAM;

	KeyFrame * newKF;

	// test for pose graph optimization
	std::set<KeyFrame *> poseGraph;

	void FindLocalGraph(KeyFrame * kf);

	// General map structure
	DeviceArray<int> heap;
	DeviceArray<int> heapCounter;
	DeviceArray<int> hashCounter;
	DeviceArray<int> bucketMutex;
	DeviceArray<Voxel> sdfBlock;
	DeviceArray<uint> noVisibleEntries;
	DeviceArray<HashEntry> hashEntries;
	DeviceArray<HashEntry> visibleEntries;

	// Used for rendering
	DeviceArray<uint> noRenderingBlocks;
	DeviceArray<RenderingBlock> renderingBlockList;
	DeviceArray2D<float> zRangeMin;
	DeviceArray2D<float> zRangeMax;
	DeviceArray2D<float> zRangeMinEnlarged;
	DeviceArray2D<float> zRangeMaxEnlarged;

	// Used for meshing
	DeviceArray<uint> nBlocks;
	DeviceArray<int3> blockPoses;
	DeviceArray<uint> noTriangles;
	DeviceArray<int> edgeTable;
	DeviceArray<int> vertexTable;
	DeviceArray2D<int> triangleTable;

	// Key Points and Re-localisation
	DeviceArray<uint> noKeys;
	DeviceArray<int> mutexKeys;
	DeviceArray<int> mapKeyIndex;
	DeviceArray<SURF> mapKeys;
	DeviceArray<SURF> tmpKeys;
	DeviceArray<SURF> surfKeys;

	//======================== REFACOTRING ========================

	DistanceField(float voxelSize, int numSdfBlock, int numHashEntry);
	DistanceField(const DistanceField&) = delete;
	DistanceField& operator=(const DistanceField&) = delete;
	~DistanceField();

	void writeMapToDisk(std::string path);
	void readMapFromDisk(std::string path);
	void copyMapDeviceToHost();
	void copyMapHostToDevice();
	void allocateHostMemory();
	void releaseHostMemory();
	void releaseDeviceMemory();

	// in between
	void allocateDeviceMemory();

protected:

	// basic map struct
	struct MapStruct
	{
		MapStruct() : memoryAllocated(false), heap(0),
				ptr_entry(0), ptr_heap(0), mutex_bucket(0),
				sdf_blocks(0), num_visible_blocks(0),
				hash_table(0), visible_entry(0) {}

		int * heap;
		int * ptr_entry;
		int * ptr_heap;
		int * mutex_bucket;
		Voxel * sdf_blocks;
		uint * num_visible_blocks;
		HashEntry * hash_table;
		HashEntry * visible_entry;
		bool memoryAllocated;
	};

	// map data stored on GPU memory.
	MapStruct device_map;
	std::mutex mutexDeviceMap;

	// map data stored on host memory i.e. RAM.
	MapStruct host_map;
	std::mutex mutexHostMap;

	// these fields should remain relatively constant
	// during one instance run, the system are not
	// designed to be able to adapt to parameter changes.
	struct MapParam
	{
		MapParam() : voxelSize(0), numBuckets(0),
				numExcessBlock(0), numSdfBlock(0),
				numEntries(0), numVoxels(0), maxNumTriangles(0),
				maxNumVertices(0), maxNumRenderingBlock(0),
				zPlaneNear(0), zPlaneFar(0), truncateDist(0) {}

		// parameters that won't affect system performance
		// too much, generally just affect the appearance
		// of the map and are free to be modified.
		// Note that due to imperfections in the function
		// PARRALLEL SCAN, too large voxelSize will not work.
		float voxelSize;

		// parameters that control the size of the
		// device memory needed in the allocation stage.
		// Note that numBuckets should always be bigger
		// than numSdfBlock as that's the requirement
		// of hash table;
		uint numBuckets;
		uint numExcessBlock;
		uint numSdfBlock;
		uint numEntries;
		uint numVoxels;
		uint maxNumTriangles;
		uint maxNumVertices;
		int maxNumRenderingBlock;

		// parameters control how far the camera sees
		// should keep them in minimum as long as they
		// satisfy your needs. Larger viewing frusta
		// will significantly slow down the system.
		// as more sdf blocks will be allocated.
		float zPlaneNear;
		float zPlaneFar;

		// constants shouldn't be changed at all
		// these are down to the basic design of the system
		// change these will render system unstable
		// ( if not unusable at all )
		const uint blockSize = 8;
		const uint blockSize3 = 512;
		float truncateDist;

	} param;

	// temporary variables
	uint numCurrentViewBlock;
	Frame * lastCheckFrame;
};

#endif

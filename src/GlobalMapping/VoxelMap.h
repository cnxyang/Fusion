#pragma once
#include "SlamSystem.h"
#include "MapStruct.h"
#include "Settings.h"
#include <opencv2/opencv.hpp>
#include <vector>

class Frame;
class KeyMap;
class SlamSystem;
class PointCloud;

class VoxelMap
{
public:

	VoxelMap();

	void resetMapStruct();
	void Release();
	void CreateModel();
	void UpdateMapKeys();
	void CreateRAM();
	void DownloadToRAM();
	void UploadFromRAM();
	void ReleaseRAM();
	bool HasNewKF();
	void FuseKeyPoints(Frame * f);
	void UpdateVisibility(Frame * f, uint & no);
	void DefuseColor(Frame * f, uint & no);
	void FuseColor(Frame * f, uint & no);
	void RayTrace(uint noVisibleBlocks, Frame * f);
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
	operator MapStruct() const;

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

public:

	VoxelMap(const VoxelMap&) = delete;
	VoxelMap& operator=(const VoxelMap&) = delete;
	~VoxelMap();

	MapStruct* device, host;
	MapState currentState;

	void writeMapToDisk(std::string path);
	void readMapFromDisk(std::string path);
	void copyMapDeviceToHost();
	void copyMapHostToDevice();
	void allocateHostMemory();
	void releaseHostMemory();
	void releaseDeviceMemory();
	void takeSnapShot(PointCloud* data, int numVisibleBlocks = -1);
	int fusePointCloud(PointCloud* data);

	void allocateDeviceMemory();

protected:

	uint updateVisibility(PointCloud* data);

	// temporary variables
	uint numCurrentViewBlock;
	Frame * lastCheckFrame;
};

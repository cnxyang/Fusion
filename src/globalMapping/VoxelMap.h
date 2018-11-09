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
	void RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float4> & nmap, float depthMin, float depthMax,
			float fx, float fy, float cx, float cy);

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

	int * mutexKeysRAM;
	SURF * mapKeysRAM;

	DeviceArray<uint> noVisibleEntries;

	// Used for rendering
	DeviceArray<uint> noRenderingBlocks;
	DeviceArray<RenderingBlock> renderingBlockList;
	DeviceArray2D<float> zRangeMin;
	DeviceArray2D<float> zRangeMax;

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

	VoxelMap(int w, int h);
	VoxelMap(const VoxelMap&) = delete;
	VoxelMap& operator=(const VoxelMap&) = delete;
	~VoxelMap();

	// Public APIS
	void allocateHostMap();
	void allocateDeviceMap();
	void releaseHostMap();
	void releaseDeviceMap();
	void copyMapToHost();
	void copyMapToDevice();
	void writeMapToDisk(const char* path);
	void readMapFromDisk(const char path);
	void exportMesh(Mesh3D* mesh);

	void raycast(PointCloud* data, int n = -1);
	int fuseImages(PointCloud* data);
	int defuseImages(PointCloud* pc);

private:

	uint updateVisibility(PointCloud* data);

	struct Data
	{
		// Used for a variety of reasons
		DeviceArray<uint> numVisibleEntries;

		// Used for rendering the synthetic view
		DeviceArray<uint> numRenderingBlocks;
		DeviceArray2D<float2> zRangeSyntheticView;
		DeviceArray2D<float2> zRangeTopdownView;
		DeviceArray<RenderingBlock> renderingBlocks;

		// Used for meshing the scene.
		DeviceArray<uint> numExistedBlocks;
		DeviceArray<int3> blockPositions;
		DeviceArray<uint> totalNumTriangle;

		// Constant look-up tables
		DeviceArray<int> constEdgeTable;
		DeviceArray<int> constVertexTable;
		DeviceArray2D<int> constTriangleTtable;

	} data;

	MapStruct* device, * host;
	int width, height;
	Eigen::Matrix3f K;
};

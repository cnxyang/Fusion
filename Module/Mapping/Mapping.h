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

class Mapping {
public:
	Mapping();

	void create();
	void reset();
	void release();
	void createModel();

	operator KeyMap() const;
	operator DeviceMap() const;

	void setTracker(Tracker * ptracker);
	void setSystem(System * psystem);

	void updateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
			float depthMin, float depthMax, float fx, float fy, float cx,
			float cy, uint & no);

	void rayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float4> & nmap, float depthMin, float depthMax,
			float fx, float fy, float cx, float cy);

	void fuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void renderMap(DeviceArray2D<float4> & vmap, DeviceArray2D<float4> & nmap,
			Matrix3f viewRot, Matrix3f viewRotInv, float3 viewTrans,
			int num_occupied_blocks);

	System * slam;
	Tracker * tracking;

	bool lost;
	std::atomic<bool> meshUpdated;
	std::atomic<bool> mapPointsUpdated;
	std::atomic<bool> mapUpdated;

	// Reconstructions
	DeviceArray<int> heap;
	DeviceArray<int> heapCounter;
	DeviceArray<int> hashCounter;
	DeviceArray<int> bucketMutex;
	DeviceArray<Voxel> sdfBlock;
	DeviceArray<uint> noVisibleEntries;
	DeviceArray<HashEntry> hashEntries;
	DeviceArray<HashEntry> visibleEntries;

	// Graph Optimisation
	std::set<KeyFrame *> keyFrames;

	// Extraction
	DeviceArray<uint> nBlocks;
	DeviceArray<float3> modelVertex;
	DeviceArray<float3> modelNormal;
	DeviceArray<uchar3> modelColor;
	DeviceArray<int3> blockPoses;
	DeviceArray<uint> noTriangles;
	DeviceArray<int> edgeTable;
	DeviceArray<int> vertexTable;
	DeviceArray2D<int> triangleTable;
	uint noTrianglesHost;

	// Rendering
	uint noBlocksInFrustum;
	DeviceArray<uint> noRenderingBlocks;
	DeviceArray<RenderingBlock> renderingBlockList;
	DeviceArray2D<float> zRangeMin;
	DeviceArray2D<float> zRangeMax;

	// Key Point
	DeviceArray<int> mKeyMutex;
	DeviceArray<SurfKey> mORBKeys;
	DeviceArray<SurfKey> tmpKeys;
	std::vector<SurfKey> hostKeys;
	uint noKeysInMap;
	bool mapKeyUpdated;
	std::vector<int> hostIndex;
	DeviceArray<int> mapIndices;
	DeviceArray<int> keyIndices;
};

#endif

#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "System.h"
#include "Tracking.h"
#include "KeyFrame.h"
#include "DeviceMap.h"

#include <mutex>
#include <vector>
#include <opencv.hpp>

struct ORBKey;
class KeyMap;
class System;
class Tracker;

class Mapping {
public:
	Mapping(bool useDefaultStream = true, cudaStream_t * stream = nullptr);
	~Mapping();


	void fuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void renderMap(DeviceArray2D<float4> & vmap, DeviceArray2D<float4> & nmap,
			Matrix3f viewRot, Matrix3f viewRotInv, float3 viewTrans,
			int num_occupied_blocks);

	void IntegrateKeys(Frame&);
	void CheckKeys(Frame& F);
	void GetORBKeys(DeviceArray<ORBKey>& keys, uint& n);
	void GetKeysHost(std::vector<ORBKey>& vkeys);

	void rayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float4> & nmap,
			float depthMin, float depthMax, float fx, float fy, float cx, float cy);

public:

	void create();
	void reset();
	void release();
	void createModel();

	operator KeyMap();
	operator DeviceMap();
	operator KeyMap() const;
	operator DeviceMap() const;

	void push_back(const KeyFrame * kf, Eigen::Matrix4d & dT);
	void push_back(KeyFrame * kf);
	void remove(const KeyFrame * kf);
	void setTracker(Tracker * ptracker);
	void setSystem(System * psystem);

	void updateKeyIndices();
	void updateMapKeys();
	void updateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
			float depthMin, float depthMax, float fx, float fy, float cx,
			float cy, uint & no);
	void fuseKeys(Frame & f, std::vector<bool> & outliers);
	std::vector<uint> getKeyIndices() const;
	std::vector<ORBKey> getAllKeys();

	Tracker * ptracker;
	System * psystem;

	std::mutex mutexMesh;
	// general control
	bool lost;
	bool useDefaultStream;
	cudaStream_t * stream;
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
	DeviceArray<ORBKey> mORBKeys;
	DeviceArray<ORBKey> tmpKeys;
	std::vector<ORBKey> hostKeys;
	uint noKeysInMap;
	bool mapKeyUpdated;
	std::vector<int> hostIndex;
	DeviceArray<int> mapIndices;
	DeviceArray<int> keyIndices;
};

#endif

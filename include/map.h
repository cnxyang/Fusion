#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "cufunc.h"
#include "devmap.h"
#include "system.h"
#include "cuarray.h"
#include "tracker.h"
#include "keyFrame.h"

#include <mutex>
#include <vector>
#include <opencv.hpp>

struct ORBKey;
class KeyMap;

class Mapping {
public:
	Mapping();
	~Mapping();


	void fuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void renderMap(DeviceArray2D<float4> & vmap, DeviceArray2D<float3> & nmap,
			Matrix3f viewRot, Matrix3f viewRotInv, float3 viewTrans,
			int num_occupied_blocks);

	void IntegrateKeys(Frame&);
	void CheckKeys(Frame& F);
	void GetORBKeys(DeviceArray<ORBKey>& keys, uint& n);
	void GetKeysHost(std::vector<ORBKey>& vkeys);

	std::vector<Eigen::Vector3d> GetCamTrace() { return mCamTrace; }

	void rayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float3> & nmap);




	std::vector<Eigen::Vector3d> mCamTrace;

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
	void fuseKeys(Frame & f, std::vector<bool> & outliers);
	std::vector<uint> getKeyIndices() const;
	std::vector<ORBKey> getAllKeys();

	Tracker * ptracker;
	System * psystem;

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
	std::mutex mutexMesh;
	DeviceArray<uint> nBlocks;
	DeviceArray<float3> modelVertex;
	DeviceArray<float3> modelNormal;
	DeviceArray<uchar3> modelColor;
	DeviceArray<int> edgeTable;
	DeviceArray<int> vertexTable;
	DeviceArray2D<int> triangleTable;
	DeviceArray<int3> blockPoses;
	DeviceArray<uint> noTriangles;
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
};

#endif

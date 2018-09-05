#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "KeyFrame.hpp"
#include "cufunc.h"
#include "cuarray.h"
#include "devmap.h"

#include <vector>
#include <opencv.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

struct ORBKey;
class KeyMap;

class Mapping {
public:
	Mapping();
	~Mapping();

	void allocate();
	void reset();
	void release();
	void createMesh();
	void fuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color, Matrix3f Rview,
			Matrix3f RviewInv, float3 tview, uint & no);

	void RenderMap(DeviceArray2D<float4> & vmap, DeviceArray2D<float3> & nmap,
			Matrix3f viewRot, Matrix3f viewRotInv, float3 viewTrans,
			int num_occupied_blocks);

	void IntegrateKeys(Frame&);
	void CheckKeys(Frame& F);
	void GetORBKeys(DeviceArray<ORBKey>& keys, uint& n);
	void GetKeysHost(std::vector<ORBKey>& vkeys);

	std::vector<Eigen::Vector3d> GetCamTrace() { return mCamTrace; }
	std::mutex mMutexMesh;

	void RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float3> & nmap);

	operator DeviceMap();
	operator const DeviceMap() const;
	operator KeyMap();
	operator const KeyMap() const;

public:

	int noBlocks;
	static bool mbFirstCall;

	DeviceArray<int> mMemory;
	DeviceArray<int> mUsedMem;
	DeviceArray<int> mEntryPtr;
	DeviceArray<int> mBucketMutex;
	DeviceArray<Voxel> mVoxelBlocks;
	DeviceArray<uint> mNumVisibleEntries;
	DeviceArray<HashEntry> mHashEntries;
	DeviceArray<HashEntry> mVisibleEntries;

	DeviceArray<int> mKeyMutex;
	DeviceArray<ORBKey> mORBKeys;

	std::vector<Eigen::Vector3d> mCamTrace;

	DeviceArray<float3> mMesh;
	DeviceArray2D<int> mTriTable;
	DeviceArray<int> mEdgeTable;
	DeviceArray<int> mNoVertex;
	DeviceArray<float3> mMeshNormal;
	DeviceArray<uchar3> mColorMap;
	DeviceArray<RenderingBlock> mRenderingBlockList;
	DeviceArray2D<float> mDepthMapMin;
	DeviceArray2D<float> mDepthMapMax;
	DeviceArray<uint> nBlocks;
	DeviceArray<uint> nTriangles;

	bool bUpdated;
	uint nTriangle;

public:

	bool lost;
	bool meshUpdated;
	bool mapPointsUpdated;
	bool mapUpdated;

	DeviceArray2D<float2> zRange;

	int noBlocksInFrustum;
	int noRenderingBlocks;

	void push_back(const KeyFrame * kf, Eigen::Matrix4d & dT);
	void push_back(const KeyFrame * kf);
	void remove(const KeyFrame * kf);

	Eigen::Matrix3f currentPose;
	std::set<KeyFrame *> keyFrames;

	bool extractColor;
	std::mutex extractionType;

	DeviceArray<int3> extractedPoses;
};

#endif

#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "Frame.hpp"
#include "device_function.hpp"
#include "device_array.hpp"
#include "device_map.hpp"

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

	void AllocateDeviceMemory();
	void ResetDeviceMemory();
	void ReleaseDeviceMemory();
	uint MeshScene();
	int FuseFrame(const Frame& frame);
	void RenderMap(Rendering& render, int num_occupied_blocks);
	uint IdentifyVisibleBlocks(const Frame& F);
	void FuseDepthAndColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color, Matrix3f viewRot,
			Matrix3f viewRotInv, float3 viewTrans, float fx, float fy, float cx,
			float cy, float depthMin, float depthMax, uint& noblock);
	void RenderMap(DeviceArray2D<float4> & vmap, DeviceArray2D<float3> & nmap,
			Matrix3f viewRot, Matrix3f viewRotInv, float3 viewTrans,
			int num_occupied_blocks);
	void IntegrateKeys(Frame&);
	void CheckKeys(Frame& F);
	void GetORBKeys(DeviceArray<ORBKey>& keys, uint& n);
	void GetKeysHost(std::vector<ORBKey>& vkeys);

	std::vector<Eigen::Vector3d> GetCamTrace() { return mCamTrace; }
	std::mutex mMutexMesh;

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
	DeviceArray<float3> mMeshNormal;
	DeviceArray<uchar3> mColorMap;
	DeviceArray<RenderingBlock> mRenderingBlockList;
	DeviceArray2D<float> mDepthMapMin;
	DeviceArray2D<float> mDepthMapMax;
	bool bUpdated;
	uint nTriangle;
};

#endif

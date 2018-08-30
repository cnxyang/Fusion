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

	void IntegrateKeys(Frame&);
	void CheckKeys(Frame& F);
	void GetORBKeys(DeviceArray<ORBKey>& keys, uint& n);
	void GetKeysHost(std::vector<ORBKey>& vkeys);

	std::vector<Eigen::Vector3d> GetCamTrace() { return mCamTrace; }

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
	uint nTriangle;
};

#endif

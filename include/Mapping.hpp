#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "Frame.hpp"
#include "device_function.hpp"
#include "device_array.hpp"
#include "device_map.hpp"

#include <vector>
#include <opencv.hpp>

struct ORBKey;
class KeyMap;

class Mapping {
public:
	Mapping();
	~Mapping();

	void AllocateDeviceMemory(MapDesc desc);
	void ResetDeviceMemory();
	void ReleaseDeviceMemory();
	int FuseFrame(const Frame& frame);
	void RenderMap(Rendering& render, int num_occupied_blocks);
	void UpdateDesc(MapDesc& desc);
	void DownloadDesc();

	void IntegrateKeys(Frame&);
	void GetORBKeys(DeviceArray<ORBKey>& keys, int& n);
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
	DeviceArray<int> mBucketMutex;
	DeviceArray<uint> mNumVisibleEntries;
	DeviceArray<HashEntry> mHashEntries;
	DeviceArray<HashEntry> mVisibleEntries;
	DeviceArray<Voxel> mVoxelBlocks;
	MapDesc mDesc;

	DeviceArray<int> mKeyMutex;
	DeviceArray<ORBKey> mORBKeys;

	std::vector<Eigen::Vector3d> mCamTrace;
};

#endif

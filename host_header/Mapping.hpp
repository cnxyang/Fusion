#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "Frame.hpp"
#include "device_function.hpp"
#include "device_array.hpp"
#include "device_struct.hpp"

#include <vector>
#include <opencv.hpp>

class Mapping {
public:
	Mapping();
	~Mapping();

	void AllocateDeviceMemory(MapDesc desc);
	void ResetDeviceMemory();
	void ReleaseDeviceMemory();
	void FuseKeyPoints(const Frame& frame);
	int FuseFrame(const Frame& frame);
	void RenderMap(Rendering& render, int num_occupied_blocks);
	void UpdateDesc(MapDesc& desc);
	void DownloadDesc();

	operator DeviceMap();
	operator const DeviceMap() const;

public:
	static bool mbFirstCall;

	DeviceArray<int> mMemory;
	DeviceArray<int> mUsedMem;
	DeviceArray<int> mBucketMutex;
	DeviceArray<int> mNumVisibleEntries;
	DeviceArray<HashEntry> mHashEntries;
	DeviceArray<HashEntry> mVisibleEntries;
	DeviceArray<Voxel> mVoxelBlocks;
	MapDesc mDesc;
};

#endif

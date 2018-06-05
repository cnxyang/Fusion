#ifndef __MAP_H__
#define __MAP_H__

#include "Frame.h"
#include "DeviceFunc.h"
#include "DeviceArray.h"
#include "DeviceStruct.h"
#include <opencv2/opencv.hpp>

class Map {
public:
	Map();
	~Map();

	void AllocateDeviceMemory(MapDesc desc);
	void ResetDeviceMemory();
	void ReleaseDeviceMemory();
	void FuseKeyPoints(const Frame& frame);
	bool MatchKeyPoint(const Frame& frame, int k);
	int FuseFrame(const Frame& frame);
	void RenderMap(Rendering& render, int num_occupied_blocks);
	void UpdateDesc(MapDesc& desc);
	void DownloadDesc();
	operator DeviceMap();
	operator const DeviceMap() const;

public:
	DeviceArray<int> mMemory;
	DeviceArray<int> mUsedMem;
	DeviceArray<int> mBucketMutex;
	DeviceArray<int> mNumVisibleEntries;
	DeviceArray<HashEntry> mHashEntries;
	DeviceArray<HashEntry> mVisibleEntries;
	DeviceArray<Voxel> mVoxelBlocks;
	MapDesc mDesc;

	DeviceArray<Point> mMapPoints;
	DeviceArray2D<char> mDescriptors;
};

#endif

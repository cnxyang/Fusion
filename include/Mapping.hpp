#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "Frame.hpp"
#include "DeviceFunc.h"
#include "DeviceArray.h"
#include "DeviceStruct.h"

#include <vector>
#include <opencv2/opencv.hpp>

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

	void SetFirstFrame(Frame& frame);
	std::vector<MapPoint> GetMapPoints() const;
	cv::cuda::GpuMat GetDescritpros() const;
	operator DeviceMap();
	operator const DeviceMap() const;

public:
	static bool mbFirstCall;
	static const int MaxNoKeyPoints = 200000;
	static const int MaxNoKeyPointsPerFrame = 1000;

	DeviceArray<int> mMemory;
	DeviceArray<int> mUsedMem;
	DeviceArray<int> mBucketMutex;
	DeviceArray<int> mNumVisibleEntries;
	DeviceArray<HashEntry> mHashEntries;
	DeviceArray<HashEntry> mVisibleEntries;
	DeviceArray<Voxel> mVoxelBlocks;
	MapDesc mDesc;

	std::vector<MapPoint> mMapPoints;
	std::vector<MapPoint> mMapPoints_new;
	cv::cuda::GpuMat mDescriptors;
	cv::cuda::GpuMat mDescriptors_new;
};

#endif
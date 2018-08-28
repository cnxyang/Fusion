#ifndef DEVICE_MAPPING_HPP__
#define DEVICE_MAPPING_HPP__

#include "Frame.hpp"
#include "device_map.hpp"
#include <opencv.hpp>
#include <cudaarithm.hpp>

void ResetKeys(KeyMap map);
void CollectKeys(KeyMap, DeviceArray<ORBKey>&, uint& n);
void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys);
void ProjectVisibleKeys(KeyMap map, Frame& F);
void BuildAdjecencyMatrix(cv::cuda::GpuMat& AM,	DeviceArray<ORBKey>& TrainKeys,
		DeviceArray<ORBKey>& QueryKeys, DeviceArray<float>& MatchDist,
		DeviceArray<ORBKey>& train_select, DeviceArray<ORBKey>& query_select,
		DeviceArray<int>& QueryIdx, DeviceArray<int>& SelectedIdx);

#endif

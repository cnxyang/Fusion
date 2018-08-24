#ifndef DEVICE_MAPPING_HPP__
#define DEVICE_MAPPING_HPP__

#include "device_map.hpp"
#include <opencv.hpp>
#include <cudaarithm.hpp>

void ResetKeys(KeyMap map);
void CollectKeys(KeyMap, DeviceArray<ORBKey>&, int&);
void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys);
void BuildAdjecencyMatrix(cv::cuda::GpuMat& AM,	DeviceArray<ORBKey>& TrainKeys,
		DeviceArray<ORBKey>& QueryKeys, DeviceArray<float>& MatchDist,
		DeviceArray<ORBKey>& train_select, DeviceArray<ORBKey>& query_select,
		DeviceArray<int>& QueryIdx, DeviceArray<int>& SelectedIdx);

#endif

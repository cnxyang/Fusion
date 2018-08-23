#ifndef DEVICE_MAPPING_HPP__
#define DEVICE_MAPPING_HPP__

#include "device_map.hpp"
#include <opencv.hpp>

void ResetKeys(KeyMap map);
void CollectKeys(KeyMap, DeviceArray<ORBKey>&, int&);
void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys);
void BuildAdjecencyMatrix(DeviceArray2D<float>& AM,	DeviceArray<ORBKey>& TrainKeys,
		DeviceArray<ORBKey>& QueryKeys, DeviceArray<float>& MatchDist);
#endif

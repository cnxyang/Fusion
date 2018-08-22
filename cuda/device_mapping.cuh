#ifndef DEVICE_MAPPING_HPP__
#define DEVICE_MAPPING_HPP__

#include "device_map.hpp"

void ResetKeys(KeyMap map);
void CollectKeys(KeyMap, DeviceArray<ORBKey>&, int&);
void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys);

void CreateBlock();

#endif

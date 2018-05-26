#ifndef __MAP_POINT_H__
#define __MAP_POINT_H__

#include <cuda_runtime.h>

class MapPoint {
public:
	MapPoint(float3& pos, int& index);

private:
	int index;
	float3 pos;
};

#endif

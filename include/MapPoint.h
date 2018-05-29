#ifndef __MAP_POINT_H__
#define __MAP_POINT_H__

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

class MapPoint {
public:
	MapPoint(cv::KeyPoint& kp, int& index);

private:
	int index;
	float3 pos;
};

#endif

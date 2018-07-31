#include "Frame.h"

class Frame;

class KeyFrame {
public:
	KeyFrame();
	KeyFrame(const Frame& frame);

public:
	std::vector<MapPoint> mMapPoints;
	cv::cuda::GpuMat mDescriptors;
	cv::Mat mRcw, mRwc;
	cv::Mat mtcw;
};

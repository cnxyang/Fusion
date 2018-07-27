#include "Frame.h"

class Frame;

class KeyFrame {
public:
	KeyFrame(const Frame& frame);

public:
	cv::Mat mRcw;
	cv::Mat mtcw;
};

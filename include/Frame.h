#ifndef __FRAME_H__
#define __FRAME_H__

#include "DeviceArray.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/cudaarithm.hpp"

class Frame {
public:
	Frame();
	Frame(const Frame& other);
	Frame(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& K);

	void ApplyIncrementTransform(cv::Mat& dRot, cv::Mat& dtrans);

	static const int numPyrs = 3;
	static cv::Mat mK[numPyrs];
	static int N[numPyrs];
	static std::pair<int, int> mPyrRes[numPyrs];

	static float fx(int pyr) { return mK[pyr].at<float>(0, 0); }
	static float fy(int pyr) { return mK[pyr].at<float>(1, 1); }
	static float cx(int pyr) { return mK[pyr].at<float>(0, 2); }
	static float cy(int pyr) { return mK[pyr].at<float>(1, 2); }

public:
	static bool mbFirstCall;
	constexpr static const float mDepthScale = 5000.0f;
	static float mDepthCutoff;
	static cv::Ptr<cv::cuda::ORB> mORB;

	DeviceArray2D<float> mDepth[numPyrs];
	DeviceArray2D<uchar> mGray[numPyrs];
	DeviceArray2D<float4> mVMap[numPyrs];
	DeviceArray2D<float3> mNMap[numPyrs];
	DeviceArray2D<char> mDescriptors;
	std::vector<cv::KeyPoint> mKeyPoints;
	cv::Mat mRcw;
	cv::Mat mRwc;
	cv::Mat mtcw;
};

#endif

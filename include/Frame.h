#ifndef __FRAME_H__
#define __FRAME_H__

#include "DeviceArray.h"
#include "DeviceStruct.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudaarithm.hpp>

class Frame {
public:
	Frame();
	~Frame();
	Frame(const Frame& other);
	Frame(const Frame& other, const Rendering& observation);
	Frame(const cv::Mat& imRGB, const cv::Mat& imD);

	void SetPose(const Frame& frame);
	void release();

	static const int numPyrs = 3;
	static cv::Mat mK[numPyrs];
	static int N[numPyrs];
	static std::pair<int, int> mPyrRes[numPyrs];

	static float fx(int pyr);
	static float fy(int pyr);
	static float cx(int pyr);
	static float cy(int pyr);
	static int cols(int pyr);
	static int rows(int pyr);
	static int pixels(int pyr);
	static void SetK(cv::Mat& K);

	operator Rendering();

public:

	static bool mbFirstCall;
	static float mDepthScale;
	static float mDepthCutoff;
	static cv::Ptr<cv::cuda::ORB> mORB;

	DeviceArray2D<float> mDepth[numPyrs];
	DeviceArray2D<uchar> mGray[numPyrs];
	DeviceArray2D<float4> mVMap[numPyrs];
	DeviceArray2D<float3> mNMap[numPyrs];
	DeviceArray2D<float> mdIx[numPyrs];
	DeviceArray2D<float> mdIy[numPyrs];

	std::vector<MapPoint> mMapPoints;
	std::vector<cv::KeyPoint> mKeyPoints;
	cv::cuda::GpuMat mDescriptors;
	cv::Mat mRcw;
	cv::Mat mRwc;
	cv::Mat mtcw;
	int mNkp;
};

#endif

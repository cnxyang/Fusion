#ifndef __CONVERTER_H__
#define __CONVERTER_H__

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#ifndef __CUDACC__
#include <sophus/se3.hpp>
#endif

class Converter {
public:
	static float3 CvMatToFloat3(const cv::Mat& mat);

#ifndef __CUDACC__
	static Eigen::Matrix<float, 4, 4> TransformToEigen(cv::Mat& r, cv::Mat& t);
	static void TransformToCv(Eigen::Matrix<float, 4, 4>& T, cv::Mat& r, cv::Mat& t);
	static std::vector<float> toQuaternion(const cv::Mat &M);
#endif
};

#endif

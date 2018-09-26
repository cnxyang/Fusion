#ifndef KEY_FRAME_HPP__
#define KEY_FRAME_HPP__

#include "Frame.h"
#include "DeviceArray.h"

#include <Eigen/Dense>
#include <opencv.hpp>

class Frame;

struct KeyFrame {

	KeyFrame();

	KeyFrame(const Frame * f);

	Eigen::Matrix3f Rotation() const;

	Eigen::Vector3f Translation() const;

	Matrix3f GpuRotation() const;

	Matrix3f GpuInvRotation() const;

	float3 GpuTranslation() const;

	Eigen::Vector3f GetWorldPoint(int i) const;

	int N;
	unsigned long frameId;

	Eigen::Matrix4f pose;

	cv::cuda::GpuMat descriptors;
	std::vector<float4> pointNormal;
	std::vector<cv::KeyPoint> keyPoints;
	std::vector<Eigen::Vector3f> mapPoints;
	std::vector<int> observations;
	mutable std::vector<int> keyIndex;

	DeviceArray2D<float> depth;
	DeviceArray2D<float4> vmap;
	DeviceArray2D<float4> nmap;
};

#endif

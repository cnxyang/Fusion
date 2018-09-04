#ifndef KEY_FRAME_HPP__
#define KEY_FRAME_HPP__

#include "Frame.hpp"
#include <Eigen/Dense>
#include <opencv.hpp>
#include "device_array.hpp"

class Frame;
struct KeyFrame;

struct SubFrame {
	Eigen::Matrix4f deltaPose;
	KeyFrame * referenceKF;
	cv::Mat scaledDepth;
	cv::Mat rawColor;
};

struct KeyFrame {

	unsigned long frameId;

	KeyFrame(const Frame * f);
	void push_back(Frame *& f);
	void find(Frame * f);
	void remove(Frame * f);

	bool valid;
	int N;
	Eigen::Matrix4d pose;
	std::set<SubFrame *> subFrames;

	DeviceArray2D<float4> vmap;
	DeviceArray2D<float3> nmap;

	cv::Mat rawColor;
	cv::Mat scaledDepth;

	std::vector<Eigen::Vector3d> frameKeys;
	cv::cuda::GpuMat frameDescriptors;
};

#endif

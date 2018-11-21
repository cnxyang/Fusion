#pragma once

#include "SophusUtil.h"
#include <vector>
#include <Eigen/Dense>
#include <cudafeatures2d.hpp>
#include <xfeatures2d/cuda.hpp>
#include <cudaarithm.hpp>

class Frame;
class MapPoint;

struct KeyPointStruct
{
	std::vector<bool> valid;
	std::vector<float> depth;
	cv::cuda::GpuMat descriptors;
	std::vector<cv::KeyPoint> keyPoints;
	std::vector<Eigen::Vector3d> pt3d;
	std::vector<int> observations;

	inline void minimizeMemoryFootprint() {
		descriptors.release();
		observations.clear();
		pt3d.clear();
		keyPoints.clear();
		depth.clear();
		valid.clear();
	}
};

class AOTracker
{
public:
	AOTracker(int w, int h, Eigen::Matrix3f K);
	SE3 trackFrame(Frame* frame, Frame* ref, int iterations);
	void extractKeyPoints(Frame* frame);
	bool trackingWasGood;

	Eigen::Matrix4d computePose(Eigen::Matrix3d matAb);

	int width, height;
	cv::cuda::SURF_CUDA SURF;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
};

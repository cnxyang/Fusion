#pragma once

#include "SophusUtil.h"
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Solver {
public:
	static bool PoseEstimate(std::vector<Eigen::Vector3d> & src,
			std::vector<Eigen::Vector3d> & ref, std::vector<bool> & outliers,
			Eigen::Matrix4d& T, int iteration, bool checkAngle = false);
};

class Frame;

class KeyPointStruct
{
	KeyPointStruct();
	SE3& pose() { return frame->pose(); }
	Frame* frame;
	std::vector<cv::KeyPoint> keyPoints;
	cv::cuda::GpuMat descriptors;
};

// Implementation of Absolute Orientation
class AOTracker
{
public:
	AOTracker(int w, int h, Eigen::Matrix3f K);
	void importReferenceFrame(Frame* frame);
	void trackReferenceFrame(Frame* frame);
	void extractKeyPoints(Frame* frame, KeyPointStruct* points);

private:

	int width, height;
	cv::Ptr<cv::BRISK> BRISKExtracter;
	cv::Ptr<cv::xfeatures2d::SURF> SURFDetector;
};

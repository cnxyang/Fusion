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
class MapPoint;

struct KeyPointStruct
{
	SE3& pose() { return frame->pose(); }
	Frame* frame;
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keyPoints;
	std::vector<MapPoint*> mapPoints;
};

// Implementation of Absolute Orientation
class AOTracker
{
public:
	AOTracker(int w, int h, Eigen::Matrix3f K);
	void importReferenceFrame(Frame* frame);
	void trackReferenceFrame(Frame* frame, Frame* ref, int iterations);
	void extractKeyPoints(Frame* frame, KeyPointStruct*& points);

private:

	int width, height;
	KeyPointStruct* referenceKP;
	cv::Ptr<cv::BRISK> BRISKExtracter;
	cv::Ptr<cv::xfeatures2d::SURF> SURFDetector;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	const int NUM_MINIMUM_MATCH = 10;
	const float RATIO_TEST_TH = 0.9;
	const float MINIMUM_INLIER_TH = 0.5;
};

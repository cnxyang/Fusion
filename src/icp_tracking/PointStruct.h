#ifndef __POINT_STRUCT__
#define __POINT_STRUCT__

#include <memory>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class PointStruct
{
public:
	PointStruct(cv::Mat &image);
	size_t detect();
	void compute();
	void draw_and_show_keypoints() const;

private:
	cv::Mat desc_;
	cv::Mat image_;
	std::vector<cv::KeyPoint> key_points_;
};

extern cv::Ptr<cv::FastFeatureDetector> fast_detector;
extern cv::Ptr<cv::xfeatures2d::SURF> surf_detector;
typedef std::shared_ptr<PointStruct> PointStructPtr;

#endif

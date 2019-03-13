#include "PointStruct.h"
#include <opencv2/opencv.hpp>

cv::Ptr<cv::FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create();
cv::Ptr<cv::xfeatures2d::SURF> surf_detector = cv::xfeatures2d::SURF::create();

PointStruct::PointStruct(cv::Mat &image) : image_(image)
{

}

size_t PointStruct::detect()
{
	fast_detector->detect(image_, key_points_);
	return key_points_.size();
}

void PointStruct::compute()
{
	surf_detector->compute(image_, key_points_, desc_);
}

void PointStruct::draw_and_show_keypoints() const
{
	if(key_points_.size() == 0)
		return;

	cv::Mat tmp_img;
	cv::drawKeypoints(image_, key_points_, tmp_img);

	cv::imshow("key points filled image", tmp_img);
	cv::waitKey(1);
}

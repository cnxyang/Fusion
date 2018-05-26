#include "Converter.h"

float3 Converter::CvMatToFloat3(const cv::Mat& mat) {
	return make_float3(mat.at<float>(0), mat.at<float>(1), mat.at<float>(2));
}

#ifndef __CUDACC__
cv::Mat Converter::EigenToCvMat(Sophus::Matrix3<double> rot) {
	cv::Mat rotd = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat rotf = cv::Mat::eye(3, 3, CV_32FC1);
	rotd.at<double>(0, 0) = rot(0, 0);
	rotd.at<double>(0, 1) = rot(0, 1);
	rotd.at<double>(0, 2) = rot(0, 2);
	rotd.at<double>(1, 0) = rot(1, 0);
	rotd.at<double>(1, 1) = rot(1, 1);
	rotd.at<double>(1, 2) = rot(1, 2);
	rotd.at<double>(2, 0) = rot(2, 0);
	rotd.at<double>(2, 1) = rot(2, 1);
	rotd.at<double>(2, 2) = rot(2, 2);
	rotd.convertTo(rotf, CV_32FC1);
	return rotf;
}

Eigen::Matrix<float, 4, 4> Converter::TransformToEigen(cv::Mat& r, cv::Mat& t) {
	Eigen::Matrix<float, 4, 4> T = Eigen::Matrix<float, 4, 4>::Identity();
	T(0, 0) = r.at<float>(0, 0);
	T(0, 1) = r.at<float>(0, 1);
	T(0, 2) = r.at<float>(0, 2);
	T(0, 3) = t.at<float>(0);
	T(1, 0) = r.at<float>(1, 0);
	T(1, 1) = r.at<float>(1, 1);
	T(1, 2) = r.at<float>(1, 2);
	T(1, 3) = t.at<float>(1);
	T(2, 0) = r.at<float>(2, 0);
	T(2, 1) = r.at<float>(2, 1);
	T(2, 2) = r.at<float>(2, 2);
	T(2, 3) = t.at<float>(2);
	return T;
}

void Converter::TransformToCv(Eigen::Matrix<float, 4, 4>& T, cv::Mat& r, cv::Mat& t) {
	r = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	r.at<float>(0, 0) = T(0, 0);
	r.at<float>(0, 1) = T(0, 1);
	r.at<float>(0, 2) = T(0, 2);
	r.at<float>(1, 0) = T(1, 0);
	r.at<float>(1, 1) = T(1, 1);
	r.at<float>(1, 2) = T(1, 2);
	r.at<float>(2, 0) = T(2, 0);
	r.at<float>(2, 1) = T(2, 1);
	r.at<float>(2, 2) = T(2, 2);
	t.at<float>(0) = T(0, 3);
	t.at<float>(1) = T(1, 3);
	t.at<float>(2) = T(2, 3);
}

#endif

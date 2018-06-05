#include "Converter.h"

float3 Converter::CvMatToFloat3(const cv::Mat& mat) {
	return make_float3(mat.at<float>(0), mat.at<float>(1), mat.at<float>(2));
}

#ifndef __CUDACC__
Eigen::Matrix<double, 4, 4> Converter::TransformToEigen(cv::Mat r, cv::Mat t) {
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T = Eigen::Matrix<double, 4, 4,  Eigen::RowMajor>::Identity();
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

void Converter::TransformToCv(Eigen::Matrix<double, 4, 4>& T, cv::Mat& r, cv::Mat& t) {
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

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

#endif

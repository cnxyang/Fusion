#include "Tracking.h"
#include "DeviceFunc.h"
#include "eigen3/Eigen/Dense"
#include "sophus/se3.hpp"
#include "Converter.h"
#include <iostream>

Tracking::Tracking() {

	mNextState = NOT_INITIALISED;
	mK = cv::Mat::eye(3, 3, CV_32FC1);
	mK.at<float>(0, 0) = 528;
	mK.at<float>(1, 1) = 528;
	mK.at<float>(0, 2) = 320;
	mK.at<float>(1, 2) = 240;
}

void Tracking::GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD) {

	mNextFrame = Frame(imRGB, imD, mK);

	Track();

	mLastFrame = Frame(mNextFrame);
}

void Tracking::Track() {
	bool bOK;
	switch(mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		bOK = TrackLastFrame();
		break;

	case LOST:
		break;
	}

	if(!bOK)
		mNextState = LOST;
}

bool Tracking::InitTracking() {
	mNextState = OK;
	return true;
}

void Tracking::VisualiseTrackingResult() {

	DeviceArray2D<uchar> diff(640, 480);
	WarpGrayScaleImage(mNextFrame, mLastFrame, diff);
	cv::Mat cvdiff(480, 640, CV_8UC1);
	diff.download((void*)cvdiff.data, cvdiff.step);
	cv::imshow("diff", cvdiff);

	cv::Mat lastgray(480, 640, CV_8UC1);
	mLastFrame.mGray[0].download((void*)lastgray.data, lastgray.step);
	cv::imshow("gray", lastgray);
	int key = cv::waitKey(10);
	if(key == 27)
		exit(0);
}

bool Tracking::TrackLastFrame() {

	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<float, 6, 6> host_a;
	Eigen::Matrix<float, 6, 1> host_b;

	mNextFrame.mRcw = mLastFrame.mRcw.clone();
	mNextFrame.mRwc = mLastFrame.mRwc.clone();
	mNextFrame.mtcw = mLastFrame.mtcw.clone();

	int iter[3] = { 10, 5, 3 };
	for(int i = 2; i >= 0; --i)
		for(int j = 0; j < iter[i]; j++) {

			ICPReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());

			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = host_b.cast<double>();

			result = dA_icp.ldlt().solve(db_icp);

			auto e = Sophus::SE3f::exp(-result.cast<float>());
			std::cout << result << std::endl;
			auto dT = e.matrix();

			Eigen::Matrix<float, 4, 4> T = Converter::TransformToEigen(mNextFrame.mRcw, mNextFrame.mtcw);
			std::cout << T << std::endl << "--------------\n";
			std::cout << dT << std::endl;
			T = dT * T;

			Converter::TransformToCv(T, mNextFrame.mRcw, mNextFrame.mtcw);
			mNextFrame.mRwc = mNextFrame.mRcw.t();
//			VisualiseTrackingResult();
	}
	return true;
}

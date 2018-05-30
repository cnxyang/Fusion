#include "Tracking.h"
#include "DeviceFunc.h"
#include "DeviceStruct.h"
#include "eigen3/Eigen/Dense"
#include "sophus/se3.hpp"
#include "Converter.h"
#include <iostream>

Tracking::Tracking() {
	mpMap = nullptr;
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

bool Tracking::TrackLastFrame() {

	TrackMap();
	TrackICP();
	return true;
}

void Tracking::TrackMap() {
//	std::vector<cv::KeyPoint>& kp = mpMap->mKeyPoints;
}

void Tracking::TrackICP() {

	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<float, 6, 6> host_a;
	Eigen::Matrix<float, 6, 1> host_b;
	mNextFrame.SetPose(mLastFrame);
	ShowResiduals();

	float cost = 0, lastcost = 0;
	int iter[3] = { 5, 3, 2 };
	for(int i = 2; i >= 0; --i)
		for(int j = 0; j < iter[i]; j++) {

			lastcost = cost;

			ICPReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data(), cost);

			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = host_b.cast<double>();

			result = dA_icp.ldlt().solve(db_icp);
			auto e = Sophus::SE3f::exp(result.cast<float>());
			auto dT = e.matrix();

			Eigen::Matrix<float, 4, 4> T = Converter::TransformToEigen(mNextFrame.mRcw, mNextFrame.mtcw);
			std::cout << "T:\n" << T << std::endl
					  << "-----------------------" << std::endl
					  << "dT:\n" << dT << std::endl;
			T = dT * T;

			Converter::TransformToCv(T, mNextFrame.mRcw, mNextFrame.mtcw);
			mNextFrame.mRwc = mNextFrame.mRcw.t();
			std::cout << "cost:\n" << cost << std::endl;
	}
	ShowResiduals();

}

void Tracking::ShowResiduals() {

	DeviceArray2D<uchar> warpImg(640, 480);
	DeviceArray2D<uchar> residual(640, 480);
	warpImg.zero();
	residual.zero();
	WarpGrayScaleImage(mNextFrame, mLastFrame, residual);
	ComputeResidualImage(residual, warpImg, mNextFrame);
	cv::Mat cvresidual(480, 640, CV_8UC1);
	warpImg.download((void*)cvresidual.data, cvresidual.step);
	cv::imshow("residual", cvresidual);

	int key = cv::waitKey(0);
	if(key == 27)
		exit(0);
}

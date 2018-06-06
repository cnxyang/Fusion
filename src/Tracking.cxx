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
	mK.at<float>(0, 0) = 528  ;
	mK.at<float>(1, 1) = 528  ;
	mK.at<float>(0, 2) = 320  ;
	mK.at<float>(1, 2) = 240  ;
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
}

void Tracking::TrackICP() {

	const float w = 0.1;
	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<float, 6, 6> host_a;
	Eigen::Matrix<float, 6, 1> host_b;
	mNextFrame.SetPose(mLastFrame);

	float cost = 0;
	int iter[3] = { 10, 5, 3 };
	for(int i = 2; i >= 0; --i)
		for(int j = 0; j < iter[i]; j++) {

			cost = ICPReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
			std::cout << "Last ICP Error: " << cost << std::endl;

			Eigen::Matrix<double, 6, 6> dA_icp = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = host_b.cast<double>();

			cost = RGBReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
			std::cout << "Last RGB Error: " << cost << std::endl;

			Eigen::Matrix<double, 6, 6> dA_rgb = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_rgb = host_b.cast<double>();

			Eigen::Matrix<double, 6, 6> dA = w * w * dA_icp + dA_rgb;
			Eigen::Matrix<double, 6, 1> db = w * db_icp + db_rgb;

			result = dA.ldlt().solve(db);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();

			Eigen::Matrix<double, 4, 4> Tc = Converter::TransformToEigen(mNextFrame.mRcw, mNextFrame.mtcw);
			Eigen::Matrix<double, 4, 4> Tp = Converter::TransformToEigen(mLastFrame.mRcw, mLastFrame.mtcw);
//			std::cout << "T:\n" << Tc << std::endl;
			Tc = Tp * (dT.inverse() * Tc.inverse() * Tp).inverse();

			Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
			mNextFrame.mRwc = mNextFrame.mRcw.t();
	}
	ShowResiduals();
}

void Tracking::AddObservation(const Rendering& render) {
	mLastFrame = Frame(mLastFrame, render);
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

//	cv::Mat hist;
//	int histSize = 256;
//	float range[] = { 0, 256 } ;
//	const float* histRange = { range };
//	cv::calcHist(&cvresidual, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
//
//	int hist_w = 512;
//	int hist_h = 400;
//	int bin_w = cvRound((double) hist_w / histSize);
//	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
//			cv::Mat());
//	for (int i = 1; i < histSize; i++) {
//		cv::line(histImage,
//				cv::Point(bin_w * (i - 1),
//						hist_h - cvRound(hist.at<float>(i - 1))),
//				cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
//				cv::Scalar(255, 0, 0), 2, 8, 0);
//	}
//	cv::imshow("histImage", histImage);
//
//	int key = cv::waitKey(10);
//	if(key == 27)
//		exit(0);
}

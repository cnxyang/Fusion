#include "Tracking.h"
#include "DeviceFunc.h"
#include "DeviceStruct.h"
#include "eigen3/Eigen/Dense"
#include "sophus/se3.hpp"
#include "Converter.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

Tracking::Tracking() {
	mpMap = nullptr;
	mNextState = NOT_INITIALISED;
	mORBMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
}

void Tracking::GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD) {

	mNextFrame = Frame(imRGB, imD);

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

	mpMap->FuseKeyPoints(mNextFrame);
	mNextState = OK;
	return true;
}

bool Tracking::TrackLastFrame() {
	mNextFrame.SetPose(mLastFrame);
//	TrackMap();
	TrackICP();
	return true;
}

#define RANSAC_MAX_ITER 100
#define RANSAC_NUM_POINTS 3
#define INLINER_THRESH 0.05

cv::Ptr<cv::cuda::ORB> ORBextractor = cv::cuda::ORB::create(1500);

bool Tracking::TrackMap() {

	std::vector<cv::KeyPoint> nextKP, lastKP;
	cv::cuda::GpuMat nextDesc, lastDesc;
	cv::cuda::GpuMat nextGray, lastGray;
	std::vector<cv::DMatch> Matches;
	nextGray.create(Frame::rows(0), Frame::cols(0), CV_8UC1);
	lastGray.create(Frame::rows(0), Frame::cols(0), CV_8UC1);

	SafeCall(	cudaMemcpy2D((void* )nextGray.data, nextGray.step,
					(void* )mNextFrame.mGray[0], mNextFrame.mGray[0].step(),
					sizeof(char) * mNextFrame.mGray[0].cols(),
					mNextFrame.mGray[0].rows(), cudaMemcpyDeviceToDevice));
	SafeCall(	cudaMemcpy2D((void* )lastGray.data, lastGray.step,
					(void* )mLastFrame.mGray[0], mLastFrame.mGray[0].step(),
					sizeof(char) * mLastFrame.mGray[0].cols(),
					mLastFrame.mGray[0].rows(), cudaMemcpyDeviceToDevice));

	ORBextractor->detectAndCompute(nextGray, cv::cuda::GpuMat(), nextKP, nextDesc);
	ORBextractor->detectAndCompute(lastGray, cv::cuda::GpuMat(), lastKP, lastDesc);
	mORBMatcher->match(nextDesc, lastDesc, Matches);

	cv::Mat nextImg, lastImg, outImg, nextD, lastD;
	nextGray.download(nextImg);
	lastGray.download(lastImg);
	nextD.create(Frame::rows(0), Frame::cols(0), CV_32FC1);
	lastD.create(Frame::rows(0), Frame::cols(0), CV_32FC1);
	mNextFrame.mDepth[0].download((void*)nextD.data, nextD.step);
	mLastFrame.mDepth[0].download((void*)lastD.data, lastD.step);
	cv::imshow("depth", nextD);
	cv::imshow("lastDepth", lastImg);

	cv::Mat outImg3;
	cv::drawMatches(nextImg, nextKP, lastImg, lastKP, Matches, outImg3);
	cv::imshow("Matches3", outImg3);

	std::vector<Eigen::Vector3d> vP, vQ;
	std::vector<cv::DMatch> newMatches;
	std::vector<cv::KeyPoint> newNextKP, newLastKP;

	int counter = 0;
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		cv::Point2f queryPt = nextKP[queryId].pt;
		cv::Point2f trainPt =  lastKP[trainId].pt;

		float nextdp = nextD.at<float>((int)(queryPt.y), (int)(queryPt.x));
		float lastdp = lastD.at<float>((int)(trainPt.y), (int)(trainPt.x));
		if(std::isnan(nextdp) || std::isnan(lastdp) || nextdp < 3e-1 || lastdp < 3e-1 || nextdp > 5 || lastdp > 5)
			continue;

		Eigen::Vector3d p, q;
		p << nextdp * (queryPt.x - Frame::cx(0)) / Frame::fx(0), nextdp * (queryPt.y - Frame::cy(0)) / Frame::fy(0), nextdp;
		q << lastdp * (trainPt.x - Frame::cx(0)) / Frame::fx(0), lastdp * (trainPt.y - Frame::cy(0)) / Frame::fy(0), lastdp;
		vP.push_back(p);
		vQ.push_back(q);
		cv::DMatch match = Matches[i];
		match.queryIdx = counter;
		match.trainIdx = counter;
		newMatches.push_back(match);
		newNextKP.push_back(nextKP[queryId]);
		newLastKP.push_back(lastKP[trainId]);
		counter++;
	}

	cv::Mat outImg2;
	cv::drawMatches(nextImg, newNextKP, lastImg, newLastKP, newMatches, outImg2);
	cv::imshow("Matches2", outImg2);

	int best_inliners = 0;
	float best_cost = 1000;
	Eigen::Matrix3d best_R;
	Eigen::Vector3d best_t;
	for (int i = 0; i < RANSAC_MAX_ITER; ++i) {

		std::vector<int> pair;
		std::vector<Eigen::Vector3d> pvec;
		std::vector<Eigen::Vector3d> qvec;
		Eigen::Vector3d p_mean, q_mean;
		p_mean = q_mean = Eigen::Vector3d::Zero();

		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			int index = std::rand() % vP.size();
			pair.push_back(index);
		}

		bool valid = true;
		for (int j = 0; j < RANSAC_NUM_POINTS; j++) {
			for (int k = 0; k < RANSAC_NUM_POINTS; ++k) {
				if (j == k)
					continue;
				if (pair[j] == pair[k])
					valid = false;
			}
		}
		if (!valid)
			continue;

		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {

			Eigen::Vector3d p, q;
			p = vP[pair[j]];
			q = vQ[pair[j]];

			p_mean += p;
			q_mean += q;

			pvec.push_back(p);
			qvec.push_back(q);
		}

		p_mean /= RANSAC_NUM_POINTS;
		q_mean /= RANSAC_NUM_POINTS;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = (V * U.transpose()).transpose();
		int detR = R.determinant();
		if (detR != 1)
			continue;

		Eigen::Vector3d t = p_mean - R * q_mean;

		int num_inliners = 0;
		Ab = Eigen::Matrix3d::Zero();
		p_mean = q_mean = Eigen::Vector3d::Zero();
		pvec.clear();
		qvec.clear();
		float cost = 0;
		for (int k = 0; k < vQ.size(); ++k) {

			Eigen::Vector3d p, q;
			p = vP[k];
			q = vQ[k];

			float dist = (p - (R * q + t)).norm();
			if (dist < INLINER_THRESH) {
				num_inliners++;
				p_mean += p;
				q_mean += q;
				pvec.push_back(p);
				qvec.push_back(q);
			}
		}

		if (num_inliners < 10)
			continue;

		if (num_inliners >= best_inliners) {
			p_mean /= num_inliners;
			q_mean /= num_inliners;
			for (int j = 0; j < num_inliners; ++j) {
				Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
			}
			best_inliners = num_inliners;
			Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Ab,	Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d V = svd2.matrixV();
			Eigen::Matrix3d U = svd2.matrixU();
			best_R = (V * U.transpose()).transpose();
			best_t = p_mean - best_R * q_mean;
//			best_R = R;
//			best_t = t;
		}
	}

	Matches.clear();
	nextKP.clear();
	lastKP.clear();
	counter = 0;
	for(int i = 0; i < vP.size(); ++i) {
		Eigen::Vector3d p = vP[i];
		Eigen::Vector3d q = vQ[i];

		float dist = (p - (best_R * q + best_t)).norm();
		if (fabs(dist) < INLINER_THRESH) {
			cv::DMatch match = newMatches[i];
			match.queryIdx = counter;
			match.trainIdx = counter;
			Matches.push_back(match);
			nextKP.push_back(newNextKP[i]);
			lastKP.push_back(newLastKP[i]);
			counter++;
		}
	}

	cv::drawMatches(nextImg, nextKP, lastImg, lastKP, Matches, outImg);
	cv::imshow("Matches", outImg);

	if(best_inliners < 10)
		return false;

	Eigen::Matrix4d Tp = Converter::TransformToEigen(mLastFrame.mRcw, mLastFrame.mtcw);
	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Td.topLeftCorner(3, 3) = best_R;
	Td.topRightCorner(3, 1) = best_t;
	std::cout << "Td: " << Tp << std::endl;

	Tc =  Td.inverse() * Tp;
	Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
	mNextFrame.mRwc = mNextFrame.mRcw.t();
}

void Tracking::TrackICP() {

	const float w = 0.1;
	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<float, 6, 6> host_a;
	Eigen::Matrix<float, 6, 1> host_b;

//	ShowResiduals();

	for(int i = 2; i >= 0; --i)
		for(int j = 0; j < iter[i]; j++) {

			cost = ICPReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
//			std::cout << "Last ICP Error: " << cost << std::endl;

			Eigen::Matrix<double, 6, 6> dA_icp = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = host_b.cast<double>();

			cost = RGBReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
//			std::cout << "Last RGB Error: " << cost << std::endl;

			Eigen::Matrix<double, 6, 6> dA_rgb = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_rgb = host_b.cast<double>();

			Eigen::Matrix<double, 6, 6> dA = w * w * dA_icp + dA_rgb;
			Eigen::Matrix<double, 6, 1> db = w * db_icp + db_rgb;
//			Eigen::Matrix<double, 6, 6> dA = dA_icp;
//			Eigen::Matrix<double, 6, 1> db = db_icp;
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

void Tracking::SetMap(Map* pMap) {
	mpMap = pMap;
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
//	int key = cv::waitKey(0);
//	if(key == 27)
//		exit(0);
//	if(key == 's') {
//		cv::imwrite("residual.jpg", cvresidual);
//	}
}

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

bool Tracking::GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD) {

	mNextFrame = Frame(imRGB, imD);

	bool bOK = Track();
	if(!bOK)
		return false;

	mLastFrame = Frame(mNextFrame);

	return true;
}

bool Tracking::Track() {
	bool bOK;
	switch(mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		bOK = TrackLastFrame();
		break;

	case LOST:
		bOK = Relocalisation();
		break;
	}

	if(!bOK) {
		mNextState = LOST;
//		Track();
	}

	mCamPos.push_back(mNextFrame.mtcw);
	return bOK;
}

bool Tracking::InitTracking() {

	mpMap->SetFirstFrame(mNextFrame);
	mNextState = OK;
	return true;
}

bool Tracking::TrackLastFrame() {
	mNextFrame.SetPose(mLastFrame);
	bool bOK = TrackMap();
	if(!bOK)
		return false;
//		TrackICP();
	return true;
}

bool Tracking::Relocalisation() {
	bool result = TrackMap();
	if(result)
		mNextState = OK;
	return result;
}

#define RANSAC_MAX_ITER 130
#define RANSAC_NUM_POINTS 5
#define INLINER_THRESH 0.01
#define HIGH_PROB_DIST 3.0

bool Tracking::TrackMap() {

	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
//	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mpMap->mDescriptors, matches, 2);
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mLastFrame.mDescriptors, matches, 2);
//	std::cout << "knn: " << matches.size() << std::endl;
	for(int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if(firstMatch.distance < 0.8 *  secondMatch.distance) {
				Matches.push_back(firstMatch);
		}
	}

	float totalDist = 0;
	for(int i = 0; i < Matches.size(); ++i) {
		totalDist += Matches[i].distance;
	}
	std::cout << "avg. dist : " << totalDist / Matches.size() << std::endl;

//	mORBMatcher->match(mNextFrame.mDescriptors, mpMap->mDescriptors, Matches);

	std::vector<Eigen::Vector3d> vNextKPs, vMapKPs;
//	std::cout << "Num:" <<  Matches.size() << std::endl;
	vNextKPs.reserve(Matches.size());
	vMapKPs.reserve(Matches.size());
	Matrix3f Rp = mNextFrame.mRcw;
	float3 tp = Converter::CvMatToFloat3(mNextFrame.mtcw);
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
//		MapPoint& trainPt = mpMap->mMapPoints[trainId];
		MapPoint& trainPt = mLastFrame.mMapPoints[trainId];

		Eigen::Vector3d p, q;
//		queryPt.pos = Rp * queryPt.pos + tp;
		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;
		vNextKPs.push_back(p);
		vMapKPs.push_back(q);
	}



	int best_inliners = 0;
	float best_cost = 1000;
	Eigen::Matrix3d best_R;
	Eigen::Vector3d best_t;
	for (int i = 0; i < RANSAC_MAX_ITER; ++i) {

		std::vector<int> pair;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			int index = std::rand() % vNextKPs.size();
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

		std::vector<Eigen::Vector3d> pvec;
		std::vector<Eigen::Vector3d> qvec;
		Eigen::Vector3d p_mean, q_mean;
		p_mean = q_mean = Eigen::Vector3d::Zero();
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {

			Eigen::Vector3d p, q;
			p = vNextKPs[pair[j]];
			q = vMapKPs[pair[j]];

			p_mean += p;
			q_mean += q;

			pvec.push_back(p);
			qvec.push_back(q);
		}

		p_mean /= RANSAC_NUM_POINTS;
		q_mean /= RANSAC_NUM_POINTS;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		float sigmap, sigmaq;
		sigmap = sigmaq = 0;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			sigmap += (pvec[j] - p_mean).norm();
			sigmaq += (qvec[j] - q_mean).norm();
			Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = U * V.transpose();
		float scale = sqrtf(sigmap / sigmaq);
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
		for (int k = 0; k < vMapKPs.size(); ++k) {

			Eigen::Vector3d p, q;
			p = vNextKPs[k];
			q = vMapKPs[k];

			float dist = (p - (R * q + t)).norm();
			if (dist < INLINER_THRESH) {
				num_inliners++;
				p_mean += p;
				q_mean += q;
				pvec.push_back(p);
				qvec.push_back(q);
			}
		}

		if (num_inliners < 0.1 * Matches.size())
			continue;

		if (num_inliners >= best_inliners) {
			p_mean /= num_inliners;
			q_mean /= num_inliners;
			sigmap = sigmaq = 0;
			for (int j = 0; j < num_inliners; ++j) {
				sigmap += (pvec[j] - p_mean).norm();
				sigmaq += (qvec[j] - q_mean).norm();
				Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
			}
			best_inliners = num_inliners;
			Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d V = svd2.matrixV();
			Eigen::Matrix3d U = svd2.matrixU();
			float scale = sqrtf(sigmap / sigmaq);
			best_R =  U * V.transpose();
			best_t = p_mean - best_R * q_mean;
//			std::cout << "scale: " << sqrtf(sigmap / sigmaq) << std::endl;
		}
	}

//	std::cout << best_inliners << std::endl;
	if(best_inliners < 0.1 * Matches.size())
		return false;

	Eigen::Matrix4d Tp = Converter::TransformToEigen(mLastFrame.mRcw, mLastFrame.mtcw);
	Eigen::Vector3d last_t = Tp.topRightCorner(3, 1);

	Tp = Converter::TransformToEigen(mLastFrame.mRcw, mLastFrame.mtcw);
	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Td.topLeftCorner(3, 3) = best_R;
	Td.topRightCorner(3, 1) = best_t;
	std::cout << "Td: " << Tp << std::endl;

	Tc =  Td.inverse() * Tp;
	Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
	mNextFrame.mRwc = mNextFrame.mRcw.t();

//	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
//	Tc.topLeftCorner(3, 3) = best_R.transpose();
//	Tc.topRightCorner(3, 1) = -best_R.transpose() * best_t;
////	std::cout << Tc << std::endl;
//
//	Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
//	mNextFrame.mRwc = mNextFrame.mRcw.t();

//	std::vector<cv::DMatch> newMatches;
//	for(int i = 0; i < Matches.size(); ++i) {
//		int queryId = Matches[i].queryIdx;
//		int trainId = Matches[i].trainIdx;
//		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
//		MapPoint& trainPt = mpMap->mMapPoints[trainId];
//
//		Eigen::Vector3d p, q;
//		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
//		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;
////
//		float dist = (p - (best_R * q + best_t)).norm();
//		if (dist < INLINER_THRESH) {
//			newMatches.push_back(Matches[i]);
//		}
//	}
//
//	FuseKeyPointsAndDescriptors(mNextFrame, mpMap->mMapPoints, mpMap->mDescriptors, newMatches);

	return true;
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

//			cost = RGBReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
//			std::cout << "Last RGB Error: " << cost << std::endl;

//			Eigen::Matrix<double, 6, 6> dA_rgb = host_a.cast<double>();
//			Eigen::Matrix<double, 6, 1> db_rgb = host_b.cast<double>();

//			Eigen::Matrix<double, 6, 6> dA = w * w * dA_icp + dA_rgb;
//			Eigen::Matrix<double, 6, 1> db = w * db_icp + db_rgb;
			Eigen::Matrix<double, 6, 6> dA = dA_icp;
			Eigen::Matrix<double, 6, 1> db = db_icp;
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

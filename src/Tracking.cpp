#include <iostream>
#include <vector>

#include "device_mapping.cuh"
#include "Tracking.hpp"
#include "Solver.hpp"
#include "Timer.hpp"
#include "sophus/se3.hpp"

using namespace cv;

Tracking::Tracking()
	:nextFrame(nullptr), lastFrame(nullptr),
	 referenceKF(nullptr), lastKF(nullptr),
	 needNewKF(false), relocKF(-5) {

	int w = 640;
	int h = 480;
	for(int i = 0; i < NUM_PYRS; ++i) {
		int cols = w / (1 << i);
		int rows = h / (1 << i);
		lastDepth[i].create(cols, rows);
		lastImage[i].create(cols, rows);
		lastVMap[i].create(cols, rows);
		lastNMap[i].create(cols, rows);
		nextDepth[i].create(cols, rows);
		nextImage[i].create(cols, rows);
		nextVMap[i].create(cols, rows);
		nextNMap[i].create(cols, rows);
		nextIdx[i].create(cols, rows);
		nextIdy[i].create(cols, rows);
	}

	depth.create(w, h);
	color.create(w, h);
	sumSE3.create(MaxThread);
	sumSO3.create(MaxThread);
	outSE3.create(1);
	outSO3.create(1);

	K = MatK(Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0));
	iteration[0] = 10;
	iteration[1] = 5;
	iteration[2] = 3;

	mNextState = NOT_INITIALISED;
	mORBMatcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
}

void Tracking::swapFrame() {
	for (int i = 0; i < NUM_PYRS; ++i) {
		nextImage[i].swap(lastImage[i]);
		nextDepth[i].swap(lastDepth[i]);
		nextVMap[i].swap(lastVMap[i]);
		nextNMap[i].swap(lastNMap[i]);
	}
}

bool Tracking::grabFrame(cv::Mat & imRGB, cv::Mat & imD) {

	swapFrame();
	color.upload((void*) imRGB.data, imRGB.step, imRGB.cols, imRGB.rows);
	ColourImageToIntensity(color, nextImage[0]);
	mNextFrame = Frame(nextImage[0], imD, referenceKF);
	return Track(imRGB, imD);
}

void Tracking::needKeyFrame() {

//	if ((referenceKF->frameId - relocKF) < 5)
//		return;

	if ((mNextFrame.frameId - referenceKF->frameId) > 0)
		needNewKF = true;
}

void Tracking::createKeyFrame(const Frame * f) {

	std::swap(lastKF, referenceKF);
	referenceKF = new KeyFrame(&mNextFrame);
	lastKF->frameDescriptors.release();
	mpMap->push_back(referenceKF);
}

void Tracking::computeICP() {

	depth.upload((void*)mNextFrame.rawDepth.data,
			mNextFrame.rawDepth.step,
			mNextFrame.rawDepth.cols,
			mNextFrame.rawDepth.rows);
	BilateralFiltering(depth, nextDepth[0], Frame::mDepthScale);

	for(int i = 1; i < NUM_PYRS; ++i) {
		PyrDownGaussian(nextDepth[i - 1], nextDepth[i]);
//		PyrDownGaussian(nextImage[i - 1], nextImage[i]);
	}

//	nextDepth[0].download((void*)nextFrame->scaledDepth, sizeof(float)*640);

	for(int i = 0; i < NUM_PYRS; ++i) {
		BackProjectPoints(nextDepth[i], nextVMap[i], Frame::mDepthCutoff,
				Frame::fx(i), Frame::fy(i), Frame::cx(i), Frame::cy(i));
		ComputeNormalMap(nextVMap[i], nextNMap[i]);
//		ComputeDerivativeImage(nextImage[i], nextIdx[i], nextIdy[i]);
	}
}

bool Tracking::computeSE3() {

	float residual[2];
	nextPose = mNextFrame.mPose;
	lastPose = mLastFrame.mPose;
	lastUpdatedPose = nextPose;
	Eigen::Matrix<double, 6, 1> vecB;
	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA;
	lastIcpError = std::numeric_limits<float>::max();
	for(int i = NUM_PYRS - 1; i >= 0; --i) {
		for(int j = 0; j < iteration[i]; ++j) {

			icpStep(nextVMap[i],
					lastVMap[i],
					nextNMap[i],
					lastNMap[i],
					sumSE3,
					outSE3,
					residual,
					matA.data(),
					vecB.data(),
					K(i),
					&mNextFrame,
					&mLastFrame);

			float icpError = sqrt(residual[0]) / residual[1];
			float icpCount = residual[1];

			if(std::isnan(icpError) || icpCount == 0)
				return false;

			result = matA.ldlt().solve(vecB);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();
			nextPose = lastPose * (dT.inverse() * nextPose.inverse() * lastPose).inverse();
			mNextFrame.SetPose(nextPose);
			lastIcpError = icpError;
		}
	}
}

bool Tracking::Track(cv::Mat& imRGB, cv::Mat& imD) {

	bool bOK;
	switch (mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		bOK = trackKeyFrame();
		computeICP();
		computeSE3();
		bOK = true;
		break;

	case LOST:
		bOK = TrackMap(true);
		break;
	}

	if (!bOK) {
		std::cout << "lost tracking" << std::endl;
		bOK = TrackMap(true);
		if(!bOK)
			SetState(LOST);
	} else {
		mLastFrame = Frame(mNextFrame);
		currentPose = nextPose;
		needKeyFrame();
		if(needNewKF)
			createKeyFrame(&mNextFrame);
		if(mNextState == OK && mLastState != LOST) {
//			mpMap->IntegrateKeys(mNextFrame);
//			mpMap->CheckKeys(mNextFrame);
		}
		SetState(OK);
		if(mLastState == LOST)
			std::cout << "Relocalisation finished in : " << mnNoAttempts << " iterations" << std::endl;
	}

	return bOK;
}

void Tracking::SetState(State s) {
	mLastState = mNextState;
	mNextState = s;
}

bool Tracking::InitTracking() {

	KeyFrame* p = referenceKF;
	referenceKF = new KeyFrame(&mNextFrame);
	delete p;
	mpMap->push_back(referenceKF);
	mNextFrame.referenceKF = referenceKF;
	computeICP();
//	mNextFrame.mOutliers.resize(mNextFrame.mNkp);
//	fill(mNextFrame.mOutliers.begin(), mNextFrame.mOutliers.end(), false);
	return true;
}

bool Tracking::TrackMap(bool bUseGraphMatching) {

	if(mLastState == OK) {
		mnNoAttempts = 0;
		mpMap->GetORBKeys(mDeviceKeys, mnMapPoints);
		desc.create(mnMapPoints, 32, CV_8UC1);
		if(mnMapPoints == 0)
			return false;

		mMapPoints.clear();
		mHostKeys.resize(mnMapPoints);
		mDeviceKeys.download((void*)mHostKeys.data(), mnMapPoints);
		for(int i = 0; i < mHostKeys.size(); ++i) {
			ORBKey& key = mHostKeys[i];
			for(int j = 0; j < 32; ++j) {
				desc.at<char>(i, j) = key.descriptor[j];
			}
			Eigen::Vector3d p;
			p << key.pos.x, key.pos.y, key.pos.z;
			mMapPoints.push_back(p);
		}
	}

	cv::cuda::GpuMat mMapDesc(desc);
	std::vector<cv::DMatch> matches;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mMapDesc, rawMatches, 2);

	for (int i = 0; i < rawMatches.size(); ++i) {
		cv::DMatch& firstMatch = rawMatches[i][0];
		cv::DMatch& secondMatch = rawMatches[i][1];
		if (firstMatch.distance < 0.85 * secondMatch.distance) {
			matches.push_back(firstMatch);
		}
		else if(bUseGraphMatching) {
			matches.push_back(firstMatch);
			matches.push_back(secondMatch);
		}
	}

	if(matches.size() < 50)
		return false;

	std::vector<Eigen::Vector3d> plist;
	std::vector<Eigen::Vector3d> qlist;

	if(bUseGraphMatching) {
		std::vector<ORBKey> vFrameKey;
		std::vector<ORBKey> vMapKey;
		std::vector<float> vDistance;
		std::vector<int> vQueryIdx;
		cv::Mat cpuFrameDesc;
		mNextFrame.mDescriptors.download(cpuFrameDesc);
		cv::Mat cpuMatching(2, matches.size(), CV_32SC1);
		for(int i = 0; i < matches.size(); ++i) {
			int trainIdx = matches[i].trainIdx;
			int queryIdx = matches[i].queryIdx;
			ORBKey trainKey = mHostKeys[trainIdx];
			ORBKey queryKey;
			if(trainKey.valid && queryKey.valid) {
				cv::Vec3f normal = mNextFrame.mNormals[queryIdx];
				Eigen::Vector3d& p = mNextFrame.mPoints[queryIdx];
				queryKey.pos = make_float3(p(0), p(1), p(2));
				queryKey.normal = make_float3(normal(0), normal(1), normal(2));
				vFrameKey.push_back(queryKey);
				vMapKey.push_back(trainKey);
				vDistance.push_back(matches[i].distance);
				vQueryIdx.push_back(queryIdx);
			}
		}

		DeviceArray<ORBKey> trainKeys(vMapKey.size());
		DeviceArray<ORBKey> queryKeys(vFrameKey.size());
		DeviceArray<float> MatchDist(vDistance.size());
		DeviceArray<int> QueryIdx(vQueryIdx.size());
		MatchDist.upload((void*)vDistance.data(), vDistance.size());
		trainKeys.upload((void*)vMapKey.data(), vMapKey.size());
		queryKeys.upload((void*)vFrameKey.data(), vFrameKey.size());
		QueryIdx.upload((void*)vQueryIdx.data(), vQueryIdx.size());
		cuda::GpuMat AdjecencyMatrix(matches.size(), matches.size(), CV_32FC1);
		DeviceArray<ORBKey> query_select, train_select;
		DeviceArray<int> SelectedIdx;
		BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist,
				train_select, query_select, QueryIdx, SelectedIdx);

		std::vector<int> vSelectedIdx;
		std::vector<ORBKey> vORB_train, vORB_query;
		vSelectedIdx.resize(SelectedIdx.size());
		vORB_train.resize(train_select.size());
		vORB_query.resize(query_select.size());
		train_select.download((void*)vORB_train.data(), vORB_train.size());
		query_select.download((void*)vORB_query.data(), vORB_query.size());
		SelectedIdx.download((void*)vSelectedIdx.data(), vSelectedIdx.size());
		for (int i = 0; i < query_select.size(); ++i) {
			Eigen::Vector3d p, q;
			if(vORB_query[i].valid &&
					vORB_train[i].valid) {
				bool redundant = false;
				for(int j = 0; j < i; j++) {
					if(vSelectedIdx[j] == vSelectedIdx[i]) {
						redundant = true;
						break;
					}
				}
				if(!redundant) {
					p << vORB_query[i].pos.x,
						 vORB_query[i].pos.y,
						 vORB_query[i].pos.z;
					q << vORB_train[i].pos.x,
						 vORB_train[i].pos.y,
						 vORB_train[i].pos.z;
					plist.push_back(p);
					qlist.push_back(q);
				}
			}
		}
	}
	else {
		for (int i = 0; i < matches.size(); ++i) {
			plist.push_back(mNextFrame.mPoints[matches[i].queryIdx]);
			qlist.push_back(mMapPoints[matches[i].trainIdx]);
		}
	}

	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(plist, qlist, mNextFrame.mOutliers, Td, 200);
	mnNoAttempts++;

	if(!bOK) {
		std::cout << "Relocalisation Failed. Attempts: " << mnNoAttempts << std::endl;
		return false;
	}

	mNextFrame.SetPose(Td.inverse());
	return true;
}

bool Tracking::TrackLastFrame() {

	mNextFrame.SetPose(mLastFrame);

	Timer::Start("Tracking", "Track Frame");
	bool bOK = TrackFrame();

	if (!bOK)
		return false;

//	Timer::Start("Tracking", "ICP");
//	bool bOK = TrackICP();
//	Timer::Stop("Tracking", "ICP");
	computeICP();
	bOK = computeSE3();
	Timer::Stop("Tracking", "Track Frame");

	return bOK;
}

bool Tracking::trackKeyFrame() {

	std::vector<cv::DMatch> refined;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, referenceKF->frameDescriptors, rawMatches, 2);

	for(int i = 0; i < rawMatches.size(); ++i) {
		if(rawMatches[i][0].distance < 0.8 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	int N = refined.size();
	if (N < 3)
		return false;

	std::vector<Eigen::Vector3d> src, ref;
	for(int i = 0; i < N; ++i) {
		src.push_back(mNextFrame.mPoints[refined[i].queryIdx]);
		ref.push_back(referenceKF->frameKeys[refined[i].trainIdx]);
	}

	Eigen::Matrix4d dT = Eigen::Matrix4d::Identity();
	bool result = Solver::SolveAbsoluteOrientation(src, ref, mNextFrame.mOutliers, dT, 100);

//	if(!result)
//		return result;

	Eigen::Matrix4d Tp = referenceKF->pose;
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Tc = dT.inverse() * Tp;
	mNextFrame.SetPose(Tc);
	std::cout << "23123" << std::endl;

	return true;
}

bool Tracking::TrackFrame() {

	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mLastFrame.mDescriptors, matches, 2);

	for (int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if (firstMatch.distance < 0.8 * secondMatch.distance) {
			Matches.push_back(firstMatch);
		}
	}

	std::vector<Eigen::Vector3d> p;
	std::vector<Eigen::Vector3d> q;
	for (int i = 0; i < Matches.size(); ++i) {
		p.push_back(mNextFrame.mPoints[Matches[i].queryIdx]);
		q.push_back(mLastFrame.mPoints[Matches[i].trainIdx]);
	}

	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(p, q, mNextFrame.mOutliers, Td, 100);

	if(!bOK) {
		Eigen::Matrix3d rot = Td.inverse().topLeftCorner(3,3);
		Eigen::Vector3d ea = rot.eulerAngles(0, 1, 2).array().sin();
		Eigen::Vector3d trans = Td.inverse().topRightCorner(3, 1);
		if(fabs(ea(0)) > mRotThresh ||
		   fabs(ea(1)) > mRotThresh ||
		   fabs(ea(2)) > mRotThresh ||
		   fabs(trans(0)) > mTransThresh ||
		   fabs(trans(1)) > mTransThresh ||
		   fabs(trans(2)) > mTransThresh) {
			std::cout << "Initial Pose Estimaton Failed." << std::endl;
			return bOK;
		}
	}

	Eigen::Matrix4d Tp = mLastFrame.mPose;
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Tc = Td.inverse() * Tp;

	mNextFrame.SetPose(Tc);

	return bOK;
}

//bool Tracking::TrackICP() {
//	float cost = Solver::SolveICP(mNextFrame, mLastFrame);
//
//	if(std::isnan(cost) || cost > 1e-3) {
//		std::cout << "Dense verification failed ." << std::endl;
//		return false;
//	}
//	return true;
//}

//void Tracking::AddObservation(const Rendering& render) {
//	mLastFrame = Frame(mLastFrame, render);
//}

void Tracking::SetMap(Mapping* pMap) {
	mpMap = pMap;
}

void Tracking::SetViewer(Viewer* pViewer) {
	mpViewer = pViewer;
}

//void Tracking::ShowResiduals() {
//
//	DeviceArray2D<uchar> warpImg(640, 480);
//	DeviceArray2D<uchar> residual(640, 480);
//	warpImg.zero();
//	residual.zero();
//	WarpGrayScaleImage(mNextFrame, mLastFrame, residual);
//	ComputeResidualImage(residual, warpImg, mNextFrame);
//	cv::Mat cvresidual(480, 640, CV_8UC1);
//	warpImg.download((void*) cvresidual.data, cvresidual.step);
//	cv::imshow("residual", cvresidual);
//}

void Tracking::ResetTracking() {
	mNextState = NOT_INITIALISED;
}

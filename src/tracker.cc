#include "timer.h"
#include "solver.h"
#include "tracker.h"
#include "sophus/se3.hpp"

using namespace cv;

Tracker::Tracker(int w, int h, float fx, float fy, float cx, float cy)
	:referenceKF(nullptr), lastKF(nullptr),
	 useIcp(true), useSo3(true), state(1), needImages(false),
	 lastState(1), lastReloc(0), imageUpdated(false) {

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
	renderedImage.create(w, h);
	renderedDepth.create(w, h);
	rgbaImage.create(w, h);

	iteration[0] = 10;
	iteration[1] = 5;
	iteration[2] = 3;

	lastIcpError = std::numeric_limits<float>::max();
	lastRgbError = std::numeric_limits<float>::max();
	lastSo3Error = std::numeric_limits<float>::max();

	K = MatK(fx, fy, cx, cy);
	orbMatcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
}

void Tracker::reset() {

	state = 1;
	lastState = 1;
	referenceKF = nullptr;
	lastKF = nullptr;
	nextPose = Eigen::Matrix4d::Identity();
	lastPose = Eigen::Matrix4d::Identity();
}

bool Tracker::track() {

	bool valid = false;

	switch(lastState) {
	case 1:
		initTracking();
		swapFrame();
		std::swap(state, lastState);
		lastState = 0;
		return true;

	case 0:
		valid = trackFrame(false);
		std::swap(state, lastState);

		if(valid) {
			lastState = 0;
			fuseMapPoint();
			swapFrame();
			return true;
		}

		lastState = -1;
		return false;

	case -1:
		std::cout << "trackingfailed" << std::endl;
		valid = relocalise();
		std::swap(state, lastState);

		if(valid) {
			std::cout << "relocalisation success" << std::endl;
			lastState = 0;
			swapFrame();
			return false;
		}

		lastState = -1;
		return false;
	}
}

void Tracker::fuseMapPoint() {
	mpMap->fuseKeys(nextFrame, outliers);
}

bool Tracker::trackFrame(bool useKF) {

	bool valid = false;
	valid = trackKeys();
	if(!valid) {
		return false;
	}

	initIcp();
	valid = computeSE3();

	return valid;
}

bool Tracker::trackKeys() {

	std::vector<cv::DMatch> refined;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	orbMatcher->knnMatch(nextFrame.descriptors, lastFrame.descriptors, rawMatches, 2);

	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.8 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	int N = refined.size();
	if (N < 3)
		return false;

	std::vector<Eigen::Vector3d> src, ref;
	for (int i = 0; i < N; ++i) {
		src.push_back(nextFrame.mPoints[refined[i].queryIdx]);
		ref.push_back(lastFrame.mPoints[refined[i].trainIdx]);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool result = Solver::SolveAbsoluteOrientation(src, ref, outliers, delta, maxIter);

	if (result) {
		lastUpdatePose = delta.inverse() * lastFrame.pose;
		nextFrame.SetPose(lastUpdatePose);
	}

	return result;
}

void Tracker::initTracking() {

	reset();
	if(useIcp)
		initIcp();
	createNewKF();
	return;
}

bool Tracker::grabFrame(const cv::Mat & image, const cv::Mat & depth) {

	color.upload((void*) image.data, image.step, image.cols, image.rows);
	ColourImageToIntensity(color, nextImage[0]);
	nextFrame = Frame(nextImage[0], depth, referenceKF);
	return track();
}

void Tracker::initIcp() {

	depth.upload((void*)nextFrame.rawDepth.data,
			nextFrame.rawDepth.step,
			nextFrame.rawDepth.cols,
			nextFrame.rawDepth.rows);
	BilateralFiltering(depth, nextDepth[0], Frame::mDepthScale);

	for(int i = 1; i < NUM_PYRS; ++i) {
		PyrDownGaussian(nextDepth[i - 1], nextDepth[i]);
		PyrDownGaussian(nextImage[i - 1], nextImage[i]);
		ResizeMap(lastVMap[i - 1], lastNMap[i - 1], lastVMap[i], lastNMap[i]);
	}

	for(int i = 0; i < NUM_PYRS; ++i) {
		BackProjectPoints(nextDepth[i], nextVMap[i], Frame::mDepthCutoff,
				Frame::fx(i), Frame::fy(i), Frame::cx(i), Frame::cy(i));
		ComputeNormalMap(nextVMap[i], nextNMap[i]);
	}
}

void Tracker::swapFrame() {

	lastFrame = Frame(nextFrame);
	if(needImages) {
		RenderImage(lastVMap[0], lastNMap[0], make_float3(0, 0, 0), renderedImage);
		depthToImage(nextDepth[0], renderedDepth);
		rgbImageToRgba(color, rgbaImage);
		imageUpdated = true;
	}
	for (int i = 0; i < NUM_PYRS; ++i) {
		nextImage[i].swap(lastImage[i]);
		nextDepth[i].swap(lastDepth[i]);
		nextVMap[i].swap(lastVMap[i]);
		nextNMap[i].swap(lastNMap[i]);
	}
}

float Tracker::rotationChanged() const {
	Eigen::Matrix4d delta = nextFrame.pose.inverse() * referenceKF->pose;
	Eigen::Matrix3d rotation = delta.topLeftCorner(3, 3);
	Eigen::Vector3d angles = rotation.eulerAngles(0, 1, 2).array().sin();
	return angles.norm();
}

float Tracker::translationChanged() const {
	Eigen::Matrix4d delta = nextFrame.pose.inverse() * referenceKF->pose;
	Eigen::Vector3d translation = delta.topRightCorner(3, 1);
	return translation.norm();
}

bool Tracker::needNewKF() {

	if(rotationChanged() >= 0.2 || translationChanged() >= 0.1)
		return true;

	return false;
}

void Tracker::createNewKF() {

	std::swap(lastKF, referenceKF);
	if(lastKF)
		lastKF->frameDescriptors.release();
	referenceKF = new KeyFrame(&nextFrame);
	mpMap->push_back(referenceKF);
}

bool Tracker::computeSE3() {

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA;
	Eigen::Matrix<double, 6, 1> vecb;
	Eigen::Matrix<double, 6, 1> result;
	lastIcpError = std::numeric_limits<float>::max();
	lastPose = lastFrame.pose;
	nextPose = nextFrame.pose;

	for(int i = NUM_PYRS - 1; i >= 0; --i) {
		for(int j = 0; j < iteration[i]; ++j) {

			float icpError = ICPReduceSum(nextVMap[i], lastVMap[i], nextNMap[i],
					lastNMap[i], nextFrame, lastFrame, i, matA.data(),
					vecb.data());

			if (std::isnan(icpError) || icpError >= 5e-4) {
				nextFrame.SetPose(lastPose);
				return false;
			}

			result = matA.ldlt().solve(vecb);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();
			nextPose = lastPose * (dT.inverse() * nextPose.inverse() * lastPose).inverse();
			nextFrame.pose = nextPose;
		}
	}

	return true;
}

bool Tracker::relocalise() {

	if(state != -1) {
		mpMap->updateMapKeys();

		if(mpMap->noKeysInMap == 0)
			return false;

		cv::Mat desc(mpMap->noKeysInMap, 32, CV_8UC1);
		mapPoints.clear();
		for(int i = 0; i < mpMap->noKeysInMap; ++i) {
			ORBKey & key = mpMap->hostKeys[i];
			for(int j = 0; j < 32; ++j) {
				desc.at<char>(i, j) = key.descriptor[j];
			}
			Eigen::Vector3d pos;
			pos << key.pos.x, key.pos.y, key.pos.z;
			mapPoints.push_back(pos);
		}
		keyDescriptors.upload(desc);
	}

	std::vector<cv::DMatch> matches;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	orbMatcher->knnMatch(nextFrame.descriptors, keyDescriptors, rawMatches, 2);

	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.85 * rawMatches[i][1].distance) {
			matches.push_back(rawMatches[i][0]);
		}
	}

	if (matches.size() < 50)
		return false;

	std::vector<Eigen::Vector3d> plist;
	std::vector<Eigen::Vector3d> qlist;
	for (int i = 0; i < matches.size(); ++i) {
		plist.push_back(nextFrame.mPoints[matches[i].queryIdx]);
		qlist.push_back(mapPoints[matches[i].trainIdx]);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(plist, qlist, outliers, delta, maxIterReloc);

	if (!bOK) {
		std::cout << "Relocalisation Failed." << std::endl;
		return false;
	}

	nextFrame.SetPose(delta.inverse());
	return true;
}

Eigen::Matrix4f Tracker::getCurrentPose() const {
	return lastFrame.pose.cast<float>();
}

//bool Tracking::TrackMap(bool bUseGraphMatching) {
//
//	if(mLastState == OK) {
//		mnNoAttempts = 0;
//		mpMap->GetORBKeys(mDeviceKeys, mnMapPoints);
//		desc.create(mnMapPoints, 32, CV_8UC1);
//		if(mnMapPoints == 0)
//			return false;
//
//		mMapPoints.clear();
//		mHostKeys.resize(mnMapPoints);
//		mDeviceKeys.download((void*)mHostKeys.data(), mnMapPoints);
//		for(int i = 0; i < mHostKeys.size(); ++i) {
//			ORBKey& key = mHostKeys[i];
//			for(int j = 0; j < 32; ++j) {
//				desc.at<char>(i, j) = key.descriptor[j];
//			}
//			Eigen::Vector3d p;
//			p << key.pos.x, key.pos.y, key.pos.z;
//			mMapPoints.push_back(p);
//		}
//	}
//
//	cv::cuda::GpuMat mMapDesc(desc);
//	std::vector<cv::DMatch> matches;
//	std::vector<std::vector<cv::DMatch>> rawMatches;
//	mORBMatcher->knnMatch(nextFrame.mDescriptors, mMapDesc, rawMatches, 2);
//
//	for (int i = 0; i < rawMatches.size(); ++i) {
//		cv::DMatch& firstMatch = rawMatches[i][0];
//		cv::DMatch& secondMatch = rawMatches[i][1];
//		if (firstMatch.distance < 0.85 * secondMatch.distance) {
//			matches.push_back(firstMatch);
//		}
//		else if(bUseGraphMatching) {
//			matches.push_back(firstMatch);
//			matches.push_back(secondMatch);
//		}
//	}
//
//	if(matches.size() < 50)
//		return false;
//
//	std::vector<Eigen::Vector3d> plist;
//	std::vector<Eigen::Vector3d> qlist;
//
//	if(bUseGraphMatching) {
//		std::vector<ORBKey> vFrameKey;
//		std::vector<ORBKey> vMapKey;
//		std::vector<float> vDistance;
//		std::vector<int> vQueryIdx;
//		cv::Mat cpuFrameDesc;
//		nextFrame.mDescriptors.download(cpuFrameDesc);
//		cv::Mat cpuMatching(2, matches.size(), CV_32SC1);
//		for(int i = 0; i < matches.size(); ++i) {
//			int trainIdx = matches[i].trainIdx;
//			int queryIdx = matches[i].queryIdx;
//			ORBKey trainKey = mHostKeys[trainIdx];
//			ORBKey queryKey;
//			if(trainKey.valid && queryKey.valid) {
//				cv::Vec3f normal = nextFrame.mNormals[queryIdx];
//				Eigen::Vector3d& p = nextFrame.mPoints[queryIdx];
//				queryKey.pos = make_float3(p(0), p(1), p(2));
//				queryKey.normal = make_float3(normal(0), normal(1), normal(2));
//				vFrameKey.push_back(queryKey);
//				vMapKey.push_back(trainKey);
//				vDistance.push_back(matches[i].distance);
//				vQueryIdx.push_back(queryIdx);
//			}
//		}
//
//		DeviceArray<ORBKey> trainKeys(vMapKey.size());
//		DeviceArray<ORBKey> queryKeys(vFrameKey.size());
//		DeviceArray<float> MatchDist(vDistance.size());
//		DeviceArray<int> QueryIdx(vQueryIdx.size());
//		MatchDist.upload((void*)vDistance.data(), vDistance.size());
//		trainKeys.upload((void*)vMapKey.data(), vMapKey.size());
//		queryKeys.upload((void*)vFrameKey.data(), vFrameKey.size());
//		QueryIdx.upload((void*)vQueryIdx.data(), vQueryIdx.size());
//		cuda::GpuMat AdjecencyMatrix(matches.size(), matches.size(), CV_32FC1);
//		DeviceArray<ORBKey> query_select, train_select;
//		DeviceArray<int> SelectedIdx;
//		BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist,
//				train_select, query_select, QueryIdx, SelectedIdx);
//
//		std::vector<int> vSelectedIdx;
//		std::vector<ORBKey> vORB_train, vORB_query;
//		vSelectedIdx.resize(SelectedIdx.size());
//		vORB_train.resize(train_select.size());
//		vORB_query.resize(query_select.size());
//		train_select.download((void*)vORB_train.data(), vORB_train.size());
//		query_select.download((void*)vORB_query.data(), vORB_query.size());
//		SelectedIdx.download((void*)vSelectedIdx.data(), vSelectedIdx.size());
//		for (int i = 0; i < query_select.size(); ++i) {
//			Eigen::Vector3d p, q;
//			if(vORB_query[i].valid &&
//					vORB_train[i].valid) {
//				bool redundant = false;
//				for(int j = 0; j < i; j++) {
//					if(vSelectedIdx[j] == vSelectedIdx[i]) {
//						redundant = true;
//						break;
//					}
//				}
//				if(!redundant) {
//					p << vORB_query[i].pos.x,
//						 vORB_query[i].pos.y,
//						 vORB_query[i].pos.z;
//					q << vORB_train[i].pos.x,
//						 vORB_train[i].pos.y,
//						 vORB_train[i].pos.z;
//					plist.push_back(p);
//					qlist.push_back(q);
//				}
//			}
//		}
//	}
//	else {
//		for (int i = 0; i < matches.size(); ++i) {
//			plist.push_back(nextFrame.mPoints[matches[i].queryIdx]);
//			qlist.push_back(mMapPoints[matches[i].trainIdx]);
//		}
//	}
//
//	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
//	bool bOK = Solver::SolveAbsoluteOrientation(plist, qlist, nextFrame.mOutliers, Td, 200);
//	mnNoAttempts++;
//
//	if(!bOK) {
//		std::cout << "Relocalisation Failed. Attempts: " << mnNoAttempts << std::endl;
//		return false;
//	}
//
//	nextFrame.SetPose(Td.inverse());
//	return true;
//}

void Tracker::setMap(Mapping* pMap) {
	mpMap = pMap;
}

void Tracker::setViewer(Viewer* pViewer) {
	mpViewer = pViewer;
}

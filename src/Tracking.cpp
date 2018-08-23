#include <iostream>
#include <vector>

#include "device_mapping.cuh"
#include "Tracking.hpp"
#include "Solver.hpp"
#include "Timer.hpp"

using namespace cv;

Tracking::Tracking():
mpMap(nullptr),
mpViewer(nullptr),
mNextState(NOT_INITIALISED) {
	mORBMatcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
}
int n = 0;
bool Tracking::Track(cv::Mat& imRGB, cv::Mat& imD) {

	Timer::Start("Tracking", "Create Frame");
	mNextFrame = Frame(imRGB, imD);
	Timer::Stop("Tracking", "Create Frame");

	mNextFrame.SetPose(Eigen::Matrix4d::Identity());

	bool bOK;
	switch (mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		if(n < 10) {
			bOK = TrackLastFrame();
			n++;
		}
		else
			bOK = TrackMap(true);
		break;

	case LOST:
		bOK = TrackMap();
		break;
	}

	if (!bOK) {
		bOK = TrackMap();
		if(!bOK)
			mNextState = LOST;
	} else {
		mLastFrame = Frame(mNextFrame);
		if(mNextState == OK)
			mpMap->IntegrateKeys(mNextFrame);
		mNextState = OK;
	}

	return bOK;
}

bool Tracking::InitTracking() {

	mNextFrame.mOutliers.resize(mNextFrame.mNkp);
	fill(mNextFrame.mOutliers.begin(), mNextFrame.mOutliers.end(), false);
	return true;
}

bool Tracking::TrackMap(bool bUseGraph) {

	mpMap->GetORBKeys(mMapPoints, mnMapPoints);
	cv::Mat desc(mnMapPoints, 32, CV_8UC1);
	if(mnMapPoints == 0)
		return false;

	vector<Eigen::Vector3d> Points;
	ORBKey* MapKeys = (ORBKey*)malloc(sizeof(ORBKey)*mnMapPoints);
	mMapPoints.download((void*)MapKeys, mnMapPoints);
	for(int i = 0; i < mnMapPoints; ++i) {
		ORBKey& key = MapKeys[i];
		for(int j = 0; j < 32; ++j) {
			desc.at<char>(i, j) = key.descriptor[j];
		}
		Eigen::Vector3d p;
		p << key.pos.x, key.pos.y, key.pos.z;
		Points.push_back(p);
	}

	cv::cuda::GpuMat mMapDesc(desc);
	std::vector<cv::DMatch> Matches;
	std::vector<cv::DMatch> matches;
	mORBMatcher->match(mNextFrame.mDescriptors, mMapDesc, matches);

	std::vector<ORBKey> mFrameKey;
	std::vector<ORBKey> mMapKey;
	std::vector<float> mDistance;
	cv::Mat cpuFrameDesc;
	mNextFrame.mDescriptors.download(cpuFrameDesc);
	cv::Mat cpuMatching(2, matches.size(), CV_32SC1);
	for(int i = 0; i < matches.size(); ++i) {
		int trainIdx = matches[i].trainIdx;
		int queryIdx = matches[i].queryIdx;
		ORBKey trainKey = MapKeys[trainIdx];
		ORBKey queryKey;
		cv::Vec3f normal = mNextFrame.mNormals[queryIdx];
		Eigen::Vector3d& p = mNextFrame.mPoints[queryIdx];
		queryKey.pos = make_float3(p(0), p(1), p(2));
		queryKey.normal = make_float3(normal(0), normal(1), normal(2));
		for(int j = 0; j < 32; ++j)
			queryKey.descriptor[j] = cpuFrameDesc.at<char>(queryIdx, j);
		mFrameKey.push_back(queryKey);
		mMapKey.push_back(trainKey);
		mDistance.push_back(matches[i].distance);
	}

	DeviceArray<ORBKey> trainKeys(mMapKey.size());
	DeviceArray<ORBKey> queryKeys(mFrameKey.size());
	DeviceArray<float> MatchDist(mDistance.size());
	MatchDist.upload((void*)mDistance.data(), mDistance.size());
	trainKeys.upload((void*)mMapKey.data(), mMapKey.size());
	queryKeys.upload((void*)mFrameKey.data(), mFrameKey.size());
	DeviceArray2D<float> AdjecencyMatrix(matches.size(), matches.size());
	BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist);

	return false;
}

bool Tracking::TrackMap() {

	Timer::Start("Tracking", "Relocalisation");
	mpMap->GetORBKeys(mMapPoints, mnMapPoints);
	cv::Mat desc(mnMapPoints, 32, CV_8UC1);
	if(mnMapPoints == 0)
		return false;

	vector<Eigen::Vector3d> Points;
	ORBKey* MapKeys = (ORBKey*)malloc(sizeof(ORBKey)*mnMapPoints);
	mMapPoints.download((void*)MapKeys, mnMapPoints);
	for(int i = 0; i < mnMapPoints; ++i) {
		ORBKey& key = MapKeys[i];
		for(int j = 0; j < 32; ++j) {
			desc.at<char>(i, j) = key.descriptor[j];
		}
		Eigen::Vector3d p;
		p << key.pos.x, key.pos.y, key.pos.z;
		Points.push_back(p);
	}

	cv::cuda::GpuMat mMapDesc(desc);
	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mMapDesc, matches, 2);

	for (int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if (firstMatch.distance < 0.85 * secondMatch.distance) {
			Matches.push_back(firstMatch);
		}
	}

	std::vector<Eigen::Vector3d> p;
	std::vector<Eigen::Vector3d> q;
	for (int i = 0; i < Matches.size(); ++i) {
		p.push_back(mNextFrame.mPoints[Matches[i].queryIdx]);
		q.push_back(Points[Matches[i].trainIdx]);
	}

	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(p, q, mNextFrame.mOutliers, Td, 200);

	if(!bOK)
		return false;

	mNextFrame.SetPose(Td.inverse());
	Timer::Stop("Tracking", "Relocalisation");

	return true;
}

void Tracking::UpdateMap() {
	mpMap->FuseFrame(mNextFrame);
}

bool Tracking::TrackLastFrame() {

	mNextFrame.SetPose(mLastFrame);

	Timer::Start("Tracking", "Track Frame");
	bool bOK = TrackFrame();
	Timer::Stop("Tracking", "Track Frame");

	if (!bOK)
		return false;

	Timer::Start("Tracking", "ICP");
	bOK = TrackICP();
	Timer::Stop("Tracking", "ICP");

	if (!bOK)
		return false;

	return true;
}

bool Tracking::TrackFrame() {

	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mLastFrame.mDescriptors,
			matches, 2);

	for (int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if (firstMatch.distance < 0.85 * secondMatch.distance) {
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
		   fabs(trans(2)) > mTransThresh)
			return false;
	}

	Eigen::Matrix4d Tp = mLastFrame.mPose;
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Tc = Td.inverse() * Tp;

	mNextFrame.SetPose(Tc);

	return true;
}

bool Tracking::TrackICP() {
	float cost = Solver::SolveICP(mNextFrame, mLastFrame);

	if(std::isnan(cost) || cost > 1e-3)
		return false;

	return true;
}

void Tracking::AddObservation(const Rendering& render) {
	mLastFrame = Frame(mLastFrame, render);
}

void Tracking::SetMap(Mapping* pMap) {
	mpMap = pMap;
}

void Tracking::SetViewer(Viewer* pViewer) {
	mpViewer = pViewer;
}

void Tracking::ShowResiduals() {

	DeviceArray2D<uchar> warpImg(640, 480);
	DeviceArray2D<uchar> residual(640, 480);
	warpImg.zero();
	residual.zero();
	WarpGrayScaleImage(mNextFrame, mLastFrame, residual);
	ComputeResidualImage(residual, warpImg, mNextFrame);
	cv::Mat cvresidual(480, 640, CV_8UC1);
	warpImg.download((void*) cvresidual.data, cvresidual.step);
	cv::imshow("residual", cvresidual);
}

void Tracking::ResetTracking() {
	mNextState = NOT_INITIALISED;
}

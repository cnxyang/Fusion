#include <iostream>
#include <vector>

#include "device_mapping.cuh"
#include "Tracking.hpp"
#include "Solver.hpp"
#include "Timer.hpp"

bool Tracking::mbTrackModel = true;

using namespace cv;

Tracking::Tracking():
mpMap(nullptr),
mpViewer(nullptr),
mNextState(NOT_INITIALISED) {

	mORBMatcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
}

bool Tracking::Track(cv::Mat& imRGB, cv::Mat& imD) {

	Timer::StartTiming("Tracking", "Create Frame");
	mNextFrame = Frame(imRGB, imD);
	Timer::StopTiming("Tracking", "Create Frame");

	mNextFrame.SetPose(Eigen::Matrix4d::Identity());

	bool bOK;
	switch (mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		bOK = TrackLastFrame();
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
		mNextState = OK;
		mLastFrame = Frame(mNextFrame);
		mpMap->IntegrateKeys(mNextFrame);
	}

	return bOK;
}

bool Tracking::InitTracking() {

	mNextFrame.mOutliers.resize(mNextFrame.mNkp);
	fill(mNextFrame.mOutliers.begin(), mNextFrame.mOutliers.end(), false);
	return true;
}

bool Tracking::TrackMap() {

	cout << "start relocalise" << endl;

	Timer::StartTiming("Tracking", "Relocalisation");
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

	if(!bOK) {
		cout << "relocalise failed." << endl;
		return false;
	}

	mNextFrame.SetPose(Td.inverse());
	Timer::StopTiming("Tracking", "Relocalisation");

	return true;
}

void Tracking::UpdateMap() {
	mpMap->FuseFrame(mNextFrame);
}

bool Tracking::TrackLastFrame() {

	mNextFrame.SetPose(mLastFrame);

	Timer::StartTiming("Tracking", "Track Frame");
	bool bOK = TrackFrame();
	Timer::StopTiming("Tracking", "Track Frame");

	if (!bOK)
		return false;

	Timer::StartTiming("Tracking", "ICP");
	bOK = TrackICP();
	Timer::StopTiming("Tracking", "ICP");

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

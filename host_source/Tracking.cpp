#include <iostream>
#include <vector>

#include "Tracking.hpp"
#include "Solver.hpp"
#include "Timer.hpp"

bool Tracking::mbTrackModel = true;

Tracking::Tracking() :
		mpMap(nullptr), mpViewer(nullptr), mNextState(NOT_INITIALISED) {

	mORBMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(
			cv::NORM_HAMMING);
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
		mNextState = LOST;
	} else {
		mNextState = OK;
	}

	Timer::StartTiming("Tracking", "Copy Frame");
	mLastFrame = Frame(mNextFrame);
	Timer::StopTiming("Tracking", "Copy Frame");

	return true;
}

bool Tracking::InitTracking() {

	mNextState = OK;
	return true;
}

bool Tracking::TrackMap() {

	return true;
}

void Tracking::UpdateMap() {
	mpMap->FuseFrame(mNextFrame);
}

void Tracking::UpdateFrame() {
}

bool Tracking::TrackLastFrame() {

	mNextFrame.SetPose(mLastFrame);

	Timer::StartTiming("Tracking", "Track Frame");
	bool bOK = TrackFrame();
	Timer::StopTiming("Tracking", "Track Frame");

	if (!bOK)
		return false;

	Timer::StartTiming("Tracking", "ICP");
	TrackICP();
	Timer::StopTiming("Tracking", "ICP");
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

	vector<bool> outliers;
	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(p, q, outliers, Td);

//	if(!bOK)
//		return false;

	Eigen::Matrix4d Tp = mLastFrame.mPose;
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Tc = Td.inverse() * Tp;
	mNextFrame.SetPose(Tc);
	return true;
}

void Tracking::TrackICP() {
	Eigen::Matrix4d Td;
	Solver::SolveICP(mNextFrame, mLastFrame, Td);
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

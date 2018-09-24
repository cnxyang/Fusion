#include "Solver.h"
#include "Tracking.h"
#include "sophus/se3.hpp"

using namespace cv;

Matrix3f eigen_to_mat3f(Eigen::Matrix3d & mat) {
	Matrix3f mat3f;
	mat3f.rowx = make_float3((float) mat(0, 0), (float) mat(0, 1), (float)mat(0, 2));
	mat3f.rowy = make_float3((float) mat(1, 0), (float) mat(1, 1), (float)mat(1, 2));
	mat3f.rowz = make_float3((float) mat(2, 0), (float) mat(2, 1), (float)mat(2, 2));
	return mat3f;
}

float3 eigen_to_float3(Eigen::Vector3d & vec) {
	return make_float3((float) vec(0), (float) vec(1), (float) vec(2));
}

Tracker::Tracker(int cols_, int rows_, float fx, float fy, float cx, float cy) {

	map = NULL;
	viewer = NULL;
	noInliers = 0;
	mappingTurnedOff = NULL;
	noMissedFrames = 0;
	state = lastState = 1;
	useGraphMatching = false;
	imageUpdated = false;
	mappingDisabled = false;
	needImages = false;
	ReferenceKF = NULL;
	LastKeyFrame = NULL;

	renderedImage.create(cols_, rows_);
	renderedDepth.create(cols_, rows_);
	rgbaImage.create(cols_, rows_);

	sumSE3.create(29, 96);
	outSE3.create(29);
	sumSO3.create(11, 96);
	outSO3.create(11);

	iteration[0] = 10;
	iteration[1] = 5;
	iteration[2] = 3;

	K = Intrinsics(fx, fy, cx, cy);
	matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

	NextFrame = new Frame();
	LastFrame = new Frame();
	NextFrame->Create(cols_, rows_);
	LastFrame->Create(cols_, rows_);
}

void Tracker::ResetTracking() {

	state = lastState = 1;
	ReferenceKF = LastKeyFrame = NULL;
	nextPose = Eigen::Matrix4d::Identity();
	lastPose = Eigen::Matrix4d::Identity();
	NextFrame->pose = nextPose;
	LastFrame->pose = lastPose;
}

bool Tracker::Track() {

	bool valid = false;
	std::swap(state, lastState);
	if(needImages) {
		if(updateImageMutex.try_lock()) {
			if(state != -1 && state != 1)
				RenderImage(LastFrame->vmap[0], LastFrame->nmap[0], make_float3(0), renderedImage);
			DepthToImage(LastFrame->depth[0], renderedDepth);
			RgbImageToRgba(LastFrame->color, rgbaImage);
			imageUpdated = true;
			updateImageMutex.unlock();
		}
	}

	switch(state) {
	case 1:
		InitTracking();
		SwapFrame();
		lastState = 0;
		return true;

	case 0:
		valid = TrackFrame();

		if(valid) {
			noMissedFrames = 0;
			if(lastState != -1) {
				lastState = 0;
				if(NeedKeyFrame())
					CreateKeyFrame();
				SwapFrame();
				return true;
			}

			lastState = 0;
			return false;
		}

		lastState = 0;
		noMissedFrames++;
		if(noMissedFrames > 10) {
			lastState = -1;
		}

		return false;

	case -1:
		valid = Relocalise();

		if(valid) {
			lastState = 0;
			SwapFrame();
			return true;
		}

		lastState = -1;
		return false;
	}
}

bool Tracker::TrackFrame() {

	bool valid = false;

//	if (!mappingDisabled) {
//		valid = TrackReferenceKF();
//	} else {
		valid = TrackLastFrame();
//	}

	if(!valid) {
		return false;
	}

	valid = ComputeSE3();
	return valid;
}

bool Tracker::TrackReferenceKF() {

	refined.clear();
	std::vector<std::vector<cv::DMatch>> rawMatches;
	matcher->knnMatch(NextFrame->descriptors, ReferenceKF->descriptors, rawMatches, 2);
	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.80 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	noInliers = refined.size();
	if (noInliers < 3)
		return false;

	refPoints.clear();
	framePoints.clear();
	for (int i = 0; i < noInliers; ++i) {
		framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
		refPoints.push_back(ReferenceKF->mapPoints[refined[i].trainIdx].cast<double>());
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	NextFrame->outliers.resize(refined.size());
	std::fill(NextFrame->outliers.begin(), NextFrame->outliers.end(), true);

	bool result = Solver::PoseEstimate(framePoints, refPoints, NextFrame->outliers, delta, maxIter, true );
	noInliers = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		NextFrame->pose = delta.inverse() * ReferenceKF->pose.cast<double>();
		for(int i = 0; i < refined.size(); ++i) {
			if(!NextFrame->outliers[i]) {
				ReferenceKF->observations[refined[i].trainIdx]++;
			}
		}
	}

	return result;
}

bool Tracker::TrackLastFrame() {

	refined.clear();
	std::vector<std::vector<cv::DMatch>> rawMatches;
	matcher->knnMatch(NextFrame->descriptors, LastFrame->descriptors, rawMatches, 2);
	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.80 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	noInliers = refined.size();
	if (noInliers < 3)
		return false;

	refPoints.clear();
	framePoints.clear();
	for (int i = 0; i < noInliers; ++i) {
		framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
		refPoints.push_back(LastFrame->mapPoints[refined[i].trainIdx].cast<double>());
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	NextFrame->outliers.resize(refined.size());
	std::fill(NextFrame->outliers.begin(), NextFrame->outliers.end(), true);

	bool result = Solver::PoseEstimate(framePoints, refPoints, NextFrame->outliers, delta, maxIter, true);
	noInliers = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		nextPose = delta.inverse() * LastFrame->pose;
		NextFrame->pose = nextPose;
	}

	return result;
}

void Tracker::FindNearestKF() {

	KeyFrame * CandidateKF = NULL;
	float norm_rot, norm_trans;
	std::set<const KeyFrame *>::iterator iter = map->keyFrames.begin();
	std::set<const KeyFrame *>::iterator lend = map->keyFrames.end();
	Eigen::Matrix4d pose = NextFrame->pose.inverse() * ReferenceKF->pose.cast<double>();
	Eigen::Matrix3d r = pose.topLeftCorner(3, 3);
	Eigen::Vector3d t = pose.topRightCorner(3, 1);
	Eigen::Vector3d angle = r.eulerAngles(0, 1, 2).array().sin();

	for(; iter != lend; ++iter) {
		const KeyFrame * kf = *iter;
		pose = NextFrame->pose.inverse() * kf->pose.cast<double>();
		r = pose.topLeftCorner(3, 3);
		t = pose.topRightCorner(3, 1);
		angle = r.eulerAngles(0, 1, 2).array().sin();

		float nrot = angle.norm();
		float ntrans = t.norm();
		if((nrot + ntrans) < (norm_rot + norm_trans)) {
			CandidateKF = const_cast<KeyFrame *>(kf);
			norm_rot = angle.norm();
			norm_trans = t.norm();
		}
	}

	if(CandidateKF != ReferenceKF) {
		ReferenceKF = CandidateKF;
	}
}

bool Tracker::NeedKeyFrame() {

	if(mappingDisabled)
		return false;

	Eigen::Matrix4f dT = NextFrame->pose.cast<float>() * ReferenceKF->pose.inverse();
	Eigen::Matrix3f dR = dT.topLeftCorner(3, 3);
	Eigen::Vector3f dt = dT.topRightCorner(3, 1);
	Eigen::Vector3f angle = dR.eulerAngles(0, 1, 2).array().sin();
	if(angle.norm() > 0.1 || dt.norm() > 0.1 )
		return true;

	return false;
}

void Tracker::CreateKeyFrame() {
	if(ReferenceKF)
		map->FuseKeyFrame(ReferenceKF);
	std::swap(ReferenceKF, LastKeyFrame);
	ReferenceKF = new KeyFrame(NextFrame);
}

void Tracker::InitTracking() {

	ResetTracking();
	CreateKeyFrame();
	return;
}

bool Tracker::GrabFrame(const cv::Mat & image, const cv::Mat & depth) {

	LastFrame->ResizeImages();
	NextFrame->ClearKeyPoints();
	NextFrame->FillImages(depth, image);
	NextFrame->ExtractKeyPoints();
	return Track();
}

void Tracker::SwapFrame() {
	std::swap(NextFrame, LastFrame);
}

bool Tracker::ComputeSE3() {

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA;
	Eigen::Matrix<double, 6, 1> vecb;
	Eigen::Matrix<double, 6, 1> result;
	lastPose = LastFrame->pose;
	nextPose = NextFrame->pose;
	Eigen::Matrix4d pose = NextFrame->pose;
	float icpError = 0;

	for(int i = Frame::NUM_PYRS - 1; i >= 0; --i) {
		for(int j = 0; j < iteration[i]; ++j) {

			ICPStep(NextFrame->vmap[i],
					LastFrame->vmap[i],
					NextFrame->nmap[i],
					LastFrame->nmap[i],
					NextFrame->GpuRotation(),
					NextFrame->GpuTranslation(),
					LastFrame->GpuRotation(),
					LastFrame->GpuInvRotation(),
					LastFrame->GpuTranslation(),
					K(i),
					sumSE3,
					outSE3,
					icpResidual,
					matA.data(),
					vecb.data());

			icpError = sqrt(icpResidual[0]) / icpResidual[1];
			int icpCount = (int) icpResidual[1];

			if (std::isnan(icpError)) {
				NextFrame->pose = lastPose;
				return false;
			}

			result = matA.ldlt().solve(vecb);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();
			nextPose = lastPose * (dT.inverse() * nextPose.inverse() * lastPose).inverse();
			NextFrame->pose = nextPose;
		}
	}

	Eigen::Matrix4d p = pose.inverse() * NextFrame->pose;
	Eigen::Matrix3d r = p.topLeftCorner(3, 3);
	Eigen::Vector3d t = p.topRightCorner(3, 1);
	Eigen::Vector3d a = r.eulerAngles(0, 1, 2).array().sin();
	if(icpError < 1e-4 || (a.norm() >= 0.1 && t.norm() >= 0.1)) {
		return true;
	}
	else {
		std::cout << icpError << " " << t.norm() << " " << a.norm() << std::endl;
		NextFrame->pose = lastPose;
		return false;
	}
}

bool Tracker::Relocalise() {

	if(lastState != -1) {

		map->UpdateMapKeys();

		if(map->noKeysHost == 0)
			return false;

		cv::Mat desc(map->noKeysHost, 64, CV_32FC1);
		mapKeys.clear();
		for(int i = 0; i < map->noKeysHost; ++i) {
			SurfKey & key = map->hostKeys[i];
			for(int j = 0; j < 64; ++j) {
				desc.at<float>(i, j) = key.descriptor[j];
			}
			Eigen::Vector3d pos;
			pos << key.pos.x, key.pos.y, key.pos.z;
			mapKeys.push_back(pos);
		}
		descriptors.upload(desc);
	}

	refined.clear();
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(NextFrame->descriptors, descriptors, matches, 2);
	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < 0.9 * matches[i][1].distance) {
			refined.push_back(matches[i][0]);
		}
	}

	if (refined.size() < 3)
		return false;

	framePoints.clear();
	refPoints.clear();
	for (int i = 0; i < refined.size(); ++i) {
		framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
		refPoints.push_back(mapKeys[refined[i].trainIdx]);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool bOK = Solver::PoseEstimate(framePoints, refPoints, outliers, delta, maxIterReloc);

	if (!bOK) {
		return false;
	}

	KeyFrame * CandidateKF = NULL;
	float norm_rot, norm_trans;
	std::set<const KeyFrame *>::iterator iter = map->keyFrames.begin();
	std::set<const KeyFrame *>::iterator lend = map->keyFrames.end();
	for(; iter != lend; ++iter) {
		const KeyFrame * kf = *iter;
		Eigen::Matrix4d pose = delta * kf->pose.cast<double>();
		Eigen::Matrix3d r = pose.topLeftCorner(3, 3);
		Eigen::Vector3d t = pose.topRightCorner(3, 1);
		Eigen::Vector3d angle = r.eulerAngles(0, 1, 2).array().sin();
		if(!CandidateKF) {
			CandidateKF = const_cast<KeyFrame *>(kf);
			norm_rot = angle.norm();
			norm_trans = t.norm();
			continue;
		}

		float nrot = angle.norm();
		float ntrans = t.norm();
		if((nrot + ntrans) < (norm_rot + norm_trans)) {
			CandidateKF = const_cast<KeyFrame *>(kf);
			norm_rot = angle.norm();
			norm_trans = t.norm();
		}
	}

	std::cout << "Reloc Sucess." << std::endl;
	NextFrame->pose = delta.inverse();
	ReferenceKF = CandidateKF;

	return true;
}

void Tracker::FilterMatching() {

//	frameKeySelected.clear();
//	mapKeySelected.clear();
//	keyDistance.clear();
//	vQueryIdx.clear();
//	cv::Mat cpuFrameDesc;
//	NextFrame->descriptors.download(cpuFrameDesc);
//	cv::Mat cpuMatching(2, refined.size(), CV_32SC1);
//
//	for (int i = 0; i < refined.size(); ++i) {
//		int trainIdx = refined[i].trainIdx;
//		int queryIdx = refined[i].queryIdx;
//		SurfKey trainKey = map->hostKeys[trainIdx];
//		SurfKey queryKey;
//		if (trainKey.valid && queryKey.valid) {
//			Eigen::Vector3f & p = NextFrame->mapPoints[queryIdx];
//			queryKey.pos = { p(0), p(1), p(2) };
//			queryKey.normal = NextFrame->pointNormal[queryIdx];
//			frameKeySelected.push_back(queryKey);
//			mapKeySelected.push_back(trainKey);
//			keyDistance.push_back(refined[i].distance);
//			vQueryIdx.push_back(queryIdx);
//		}
//	}
//
//	DeviceArray<SurfKey> trainKeys(mapKeySelected.size());
//	DeviceArray<SurfKey> queryKeys(frameKeySelected.size());
//	DeviceArray<float> MatchDist(keyDistance.size());
//	DeviceArray<int> QueryIdx(vQueryIdx.size());
//
//	MatchDist.upload((void*) keyDistance.data(), keyDistance.size());
//	trainKeys.upload((void*) mapKeySelected.data(), mapKeySelected.size());
//	queryKeys.upload((void*) frameKeySelected.data(), frameKeySelected.size());
//	QueryIdx.upload((void*) vQueryIdx.data(), vQueryIdx.size());
//
//	cuda::GpuMat AdjecencyMatrix(refined.size(), refined.size(), CV_32FC1);
//	DeviceArray<SurfKey> query_select, train_select;
//	DeviceArray<int> SelectedIdx;
//
//	BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist,
//			train_select, query_select, QueryIdx, SelectedIdx);
//
//	std::vector<int> vSelectedIdx;
//	std::vector<SurfKey> vORB_train, vORB_query;
//	vSelectedIdx.resize(SelectedIdx.size);
//	vORB_train.resize(train_select.size);
//	vORB_query.resize(query_select.size);
//
//	train_select.download((void*) vORB_train.data(), vORB_train.size());
//	query_select.download((void*) vORB_query.data(), vORB_query.size());
//	SelectedIdx.download((void*) vSelectedIdx.data(), vSelectedIdx.size());
//
//	for (int i = 0; i < query_select.size; ++i) {
//		Eigen::Vector3d p, q;
//		if (vORB_query[i].valid && vORB_train[i].valid) {
//			bool redundant = false;
//			for (int j = 0; j < i; j++) {
//				if (vSelectedIdx[j] == vSelectedIdx[i]) {
//					redundant = true;
//					break;
//				}
//			}
//			if (!redundant) {
//				p << vORB_query[i].pos.x, vORB_query[i].pos.y, vORB_query[i].pos.z;
//				q << vORB_train[i].pos.x, vORB_train[i].pos.y, vORB_train[i].pos.z;
//				plist.push_back(p);
//				qlist.push_back(q);
//			}
//		}
//	}
}

Eigen::Matrix4f Tracker::GetCurrentPose() const {
	return LastFrame->pose.cast<float>();
}

void Tracker::SetMap(Mapping* pMap) {
	map = pMap;
}

void Tracker::SetViewer(Viewer* pViewer) {
	viewer = pViewer;
}

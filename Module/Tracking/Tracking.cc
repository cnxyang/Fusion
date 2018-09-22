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

Tracker::Tracker(int cols_, int rows_, float fx, float fy, float cx, float cy) :
		useGraphMatching(false), state(1), needImages(false), lastState(1), imageUpdated(false), mappingDisabled(false) {

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
	lastIcpError = std::numeric_limits<float>::max();
	lastSo3Error = std::numeric_limits<float>::max();
	matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

	NextFrame = new Frame();
	LastFrame = new Frame();
	NextFrame->Create(cols_, rows_);
	LastFrame->Create(cols_, rows_);
}

void Tracker::ResetTracking() {

	state = 1;
	lastState = 1;
	nextPose = Eigen::Matrix4d::Identity();
	lastPose = Eigen::Matrix4d::Identity();
}

bool Tracker::Track() {

	bool valid = false;
	std::swap(state, lastState);

	if(needImages) {
		if(state != -1)
			RenderImage(LastFrame->vmap[0], LastFrame->nmap[0], make_float3(0, 0, 0), renderedImage);
		DepthToImage(LastFrame->depth[0], renderedDepth);
		RgbImageToRgba(LastFrame->color, rgbaImage);
		imageUpdated = true;
	}

	switch(state) {
	case 1:
		InitTracking();
		SwapFrame();
		lastState = 0;
		return true;

	case 0:
		valid = TrackFrame();
		lastState = 0;

		if(valid) {
			noMissedFrames = 0;
			if(state != -1) {
				SwapFrame();
				return true;
			}
			return false;
		}

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
			return false;
		}

		lastState = -1;
		return false;
	}
}

bool Tracker::TrackFrame() {

	bool valid = false;
	valid = TrackLastFrame();
	if(!valid) {
		return false;
	}

	valid = ComputeSE3();
	return valid;
}

bool Tracker::TrackLastFrame() {

	std::vector<cv::DMatch> refined;
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

	std::vector<Eigen::Vector3d> src, ref;
	for (int i = 0; i < noInliers; ++i) {
		Eigen::Vector3d p0, p1;
		float3 p2 = NextFrame->pt3d[refined[i].queryIdx];
		float3 p3 = LastFrame->pt3d[refined[i].trainIdx];
		p0 << p2.x, p2.y, p2.z;
		p1 << p3.x, p3.y, p3.z;
		src.push_back(p0);
		ref.push_back(p1);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool result = Solver::PoseEstimate(src, ref, outliers, delta, maxIter);

	outliers.resize(refined.size());
	noInliers = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		nextPose = delta.inverse() * LastFrame->pose;
		NextFrame->pose = nextPose;
	}

	return result;
}

void Tracker::InitTracking() {

	ResetTracking();
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
	lastIcpError = std::numeric_limits<float>::max();
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
					NextFrame->Rot_gpu(),
					NextFrame->Trans_gpu(),
					LastFrame->Rot_gpu(),
					LastFrame->RotInv_gpu(),
					LastFrame->Trans_gpu(),
					K(i),
					sumSE3,
					outSE3,
					icpResidual,
					matA.data(),
					vecb.data());

			icpError = sqrt(icpResidual[0]) / icpResidual[1];
			int icpCount = (int) icpResidual[1];

			if (std::isnan(icpError)) {
				NextFrame->SetPose(lastPose);
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
	if(icpError >= 5e-4 || a.norm() >= 0.2 || t.norm() >= 0.2) {
		std::cout << icpError << " " << t.norm() << " " << a.norm() << std::endl;
		NextFrame->pose = lastPose;
		return false;
	}
	else
		return true;
}

bool Tracker::Relocalise() {

//	if(lastState != -1) {
//		mpMap->updateMapKeys();
//
//		if(mpMap->noKeysInMap == 0)
//			return false;
//
//		cv::Mat desc(mpMap->noKeysInMap, 32, CV_8UC1);
//		mapPoints.clear();
//		for(int i = 0; i < mpMap->noKeysInMap; ++i) {
//			ORBKey & key = mpMap->hostKeys[i];
//			for(int j = 0; j < 32; ++j) {
//				desc.at<char>(i, j) = key.descriptor[j];
//			}
//			Eigen::Vector3d pos;
//			pos << key.pos.x, key.pos.y, key.pos.z;
//			mapPoints.push_back(pos);
//		}
//		keyDescriptors.upload(desc);
//	}
//
//	std::vector<cv::DMatch> matches;
//	std::vector<std::vector<cv::DMatch>> rawMatches;
//	orbMatcher->knnMatch(nextFrame.descriptors, keyDescriptors, rawMatches, 2);
//
//	for (int i = 0; i < rawMatches.size(); ++i) {
//		if (rawMatches[i][0].distance < 0.85 * rawMatches[i][1].distance) {
//			matches.push_back(rawMatches[i][0]);
//		}
//		else if (graphMatching) {
//			matches.push_back(rawMatches[i][0]);
//			matches.push_back(rawMatches[i][1]);
//		}
//	}
//
//	if (matches.size() < 50)
//		return false;
//
//	std::vector<Eigen::Vector3d> plist;
//	std::vector<Eigen::Vector3d> qlist;
//	if (graphMatching) {
//		std::vector<ORBKey> vFrameKey;
//		std::vector<ORBKey> vMapKey;
//		std::vector<float> vDistance;
//		std::vector<int> vQueryIdx;
//		cv::Mat cpuFrameDesc;
//		nextFrame.descriptors.download(cpuFrameDesc);
//		cv::Mat cpuMatching(2, matches.size(), CV_32SC1);
//
//		for (int i = 0; i < matches.size(); ++i) {
//			int trainIdx = matches[i].trainIdx;
//			int queryIdx = matches[i].queryIdx;
//			ORBKey trainKey = mpMap->hostKeys[trainIdx];
//			ORBKey queryKey;
//			if (trainKey.valid && queryKey.valid) {
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
//
//		MatchDist.upload((void*) vDistance.data(), vDistance.size());
//		trainKeys.upload((void*) vMapKey.data(), vMapKey.size());
//		queryKeys.upload((void*) vFrameKey.data(), vFrameKey.size());
//		QueryIdx.upload((void*) vQueryIdx.data(), vQueryIdx.size());
//
//		cuda::GpuMat AdjecencyMatrix(matches.size(), matches.size(), CV_32FC1);
//		DeviceArray<ORBKey> query_select, train_select;
//		DeviceArray<int> SelectedIdx;
//
//		BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist,
//				train_select, query_select, QueryIdx, SelectedIdx);
//
//		std::vector<int> vSelectedIdx;
//		std::vector<ORBKey> vORB_train, vORB_query;
//		vSelectedIdx.resize(SelectedIdx.size);
//		vORB_train.resize(train_select.size);
//		vORB_query.resize(query_select.size);
//
//		train_select.download((void*) vORB_train.data(), vORB_train.size());
//		query_select.download((void*) vORB_query.data(), vORB_query.size());
//		SelectedIdx.download((void*) vSelectedIdx.data(), vSelectedIdx.size());
//
//		for (int i = 0; i < query_select.size; ++i) {
//			Eigen::Vector3d p, q;
//			if (vORB_query[i].valid && vORB_train[i].valid) {
//				bool redundant = false;
//				for (int j = 0; j < i; j++) {
//					if (vSelectedIdx[j] == vSelectedIdx[i]) {
//						redundant = true;
//						break;
//					}
//				}
//				if (!redundant) {
//					p << vORB_query[i].pos.x, vORB_query[i].pos.y, vORB_query[i].pos.z;
//					q << vORB_train[i].pos.x, vORB_train[i].pos.y, vORB_train[i].pos.z;
//					plist.push_back(p);
//					qlist.push_back(q);
//				}
//			}
//		}
//	}
//	else {
//		for (int i = 0; i < matches.size(); ++i) {
//			plist.push_back(nextFrame.mPoints[matches[i].queryIdx]);
//			qlist.push_back(mapPoints[matches[i].trainIdx]);
//		}
//	}
//
//	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
//	bool bOK = Solver::PoseEstimate(plist, qlist, outliers, delta, maxIterReloc);
//
//	if (!bOK) {
//		return false;
//	}
//
//	nextFrame.index.resize(nextFrame.N);
//	fill(nextFrame.index.begin(), nextFrame.index.end(), -1);
//	for(int i = 0; i < matches.size(); ++i) {
//		if(!outliers[i]) {
//			nextFrame.index[matches[i].queryIdx] = mpMap->hostIndex[matches[i].trainIdx];
//		}
//	}
//	nextFrame.SetPose(delta.inverse());
//	return true;
	return true;
}

Eigen::Matrix4f Tracker::GetCurrentPose() const {
	return LastFrame->pose.cast<float>();
}

void Tracker::SetMap(Mapping* pMap) {
	mpMap = pMap;
}

void Tracker::SetViewer(Viewer* pViewer) {
	mpViewer = pViewer;
}

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
		map(NULL), viewer(NULL), noInliers(0), mappingTurnedOff(NULL),
		state(1), lastState(1), noMissedFrames(0), useGraphMatching(false),
		imageUpdated(false), mappingDisabled(false), needImages(false),
		ReferenceKF(NULL), LastKeyFrame(NULL) {

	renderedImage.create(cols_, rows_);
	renderedDepth.create(cols_, rows_);
	rgbaImage.create(cols_, rows_);

	sumSE3.create(29, 96);
	outSE3.create(29);
	sumSO3.create(11, 96);
	outSO3.create(11);
	sumRes.create(2, 96);
	outRes.create(2);

	iteration[0] = 10;
	iteration[1] = 5;
	iteration[2] = 3;

	minIcpCount[0] = 0;
	minIcpCount[1] = 0;
	minIcpCount[2] = 0;

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

//-----------------------------------------
// Main Control Flow
//-----------------------------------------
bool Tracker::Track() {

	bool valid = false;
	std::swap(state, lastState);
	if(needImages) {
		RenderView();
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
				else
					CheckOutliers();
				SwapFrame();
				return true;
			}

			lastState = 0;
			return false;
		}

		lastState = 0;
		noMissedFrames++;
		if(noMissedFrames > 9) {
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

void Tracker::CheckOutliers() {

	// to prevent system from crushing
	// when have loaded a pre-built map
	// TODO: get rid of this dirty hack
	if(!ReferenceKF)
		return;

	Eigen::Matrix4f deltaT = ReferenceKF->pose.inverse() * NextFrame->pose.cast<float>();
	Eigen::Matrix3f deltaR = deltaT.topLeftCorner(3, 3);
	Eigen::Vector3f deltat = deltaT.topRightCorner(3, 1);

	std::vector<cv::DMatch> matches;
	matcher->match(NextFrame->descriptors, ReferenceKF->descriptors, matches);
	for(int i = 0; i < matches.size(); ++i) {
		Eigen::Vector3f src = NextFrame->mapPoints[matches[i].queryIdx];
		Eigen::Vector3f ref = ReferenceKF->mapPoints[matches[i].trainIdx];
		double d = (src - (deltaR * ref + deltat)).norm();
		if (d <= 0.05f) {
			ReferenceKF->observations[matches[i].trainIdx]++;
		}
	}
}

void Tracker::RenderView() {

	if (updateImageMutex.try_lock()) {
		if (state == 0 && lastState != -1)
			RenderImage(LastFrame->vmap[0], LastFrame->nmap[0], make_float3(0), renderedImage);
		DepthToImage(LastFrame->range, renderedDepth);
		RgbImageToRgba(LastFrame->color, rgbaImage);
		imageUpdated = true;
		updateImageMutex.unlock();
	}
}

bool Tracker::TrackFrame() {

	bool valid = false;
	NextFrame->pose = LastFrame->pose;

	valid = TrackLastFrame();

	if(!valid) {
		std::cout << "track last frame failed" << std::endl;
		return false;
	}

//	ComputeSO3();
	valid = ComputeSE3();
	return valid;
}

//-----------------------------------------
// Track Current Frame w.r.t Key Frame
// TODO: deprecated due to lack of effective
// key frame management, will investigate later
//-----------------------------------------
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

//-----------------------------------------
// Track Current Frame w.r.t Last Frame
//-----------------------------------------
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

bool Tracker::TrackLastFrame_g2o() {

	refPoints.clear();
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(NextFrame->descriptors, LastFrame->descriptors, matches, 2);
	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < 0.95 * matches[i][1].distance) {
			refined.push_back(matches[i][0]);
		}
	}

	for(int i = 0; i < refined.size(); ++i) {
		refPoints.push_back(LastFrame->mapPoints[refined[i].trainIdx].cast<double>());
	}

	Eigen::Matrix4d dt;
	NextFrame->outliers.resize(NextFrame->N);
	std::fill(NextFrame->outliers.begin(), NextFrame->outliers.end(), false);
	int n = Optimizer::OptimizePose(NextFrame, refPoints, dt);

	NextFrame->pose = dt.inverse() * LastFrame->pose;

	std::cout << refined.size() << std::endl;
	return true;
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

	for (; iter != lend; ++iter) {
		const KeyFrame * kf = *iter;
		pose = NextFrame->pose.inverse() * kf->pose.cast<double>();
		r = pose.topLeftCorner(3, 3);
		t = pose.topRightCorner(3, 1);
		angle = r.eulerAngles(0, 1, 2).array().sin();

		float nrot = angle.norm();
		float ntrans = t.norm();
		if ((nrot + ntrans) < (norm_rot + norm_trans)) {
			CandidateKF = const_cast<KeyFrame *>(kf);
			norm_rot = angle.norm();
			norm_trans = t.norm();
		}
	}

	if (CandidateKF != ReferenceKF) {
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
	if (angle.norm() > 0.1 || dt.norm() > 0.1)
		return true;

	return false;
}

void Tracker::CreateKeyFrame() {

	if (ReferenceKF)
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

void Tracker::ComputeSO3() {

	Eigen::Matrix<double, 3, 3, Eigen::RowMajorBit> matA;
	Eigen::Vector3d vecb;
	Eigen::Vector3d result;
	Eigen::Matrix3d lastRot = LastFrame->Rotation();
	Eigen::Matrix3d nextRot = NextFrame->Rotation();
	Eigen::Matrix3d rot = nextRot;
	Eigen::Matrix3d delta = Eigen::Matrix3d::Identity();

	float so3Error = 0;
	int so3Count = 0;
	int pyrLevel = 2;
	float lastSO3Error = std::numeric_limits<float>::max();

	for(int i = 0; i < 10; ++i) {

		SO3Step(NextFrame->image[pyrLevel],
				LastFrame->image[pyrLevel],
				NextFrame->dIdx[pyrLevel],
				NextFrame->dIdy[pyrLevel],
				NextFrame->GpuInvRotation(),
				LastFrame->GpuRotation(),
				K(pyrLevel),
				sumSO3,
				outSO3,
				so3Residual,
				matA.data(),
				vecb.data());

		so3Error = sqrt(so3Residual[0]) / so3Residual[1];
		so3Count = (int) so3Residual[1];

		if(lastSO3Error < so3Error)
			return;

		result = matA.ldlt().solve(vecb);
		auto e = Sophus::SO3d::exp(result);
		auto dT = e.matrix();

		delta = dT * delta;
		nextRot = lastRot * delta.inverse();
		NextFrame->pose.topLeftCorner(3, 3) = nextRot;
		lastSO3Error = so3Error;
	}
}

bool Tracker::ComputeSE3() {

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA_icp;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA_rgb;
	Eigen::Matrix<double, 6, 1> vecb;
	Eigen::Matrix<double, 6, 1> vecb_icp;
	Eigen::Matrix<double, 6, 1> vecb_rgb;
	Eigen::Matrix<double, 6, 1> result;
	lastPose = LastFrame->pose;
	nextPose = NextFrame->pose;
	Eigen::Matrix4d pose = NextFrame->pose;

	float rgbError = 0;
	float icpError = 0;
	int rgbCount = 0;
	int icpCount = 0;

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();

	for(int i = Frame::NUM_PYRS - 1; i >= 0; --i) {
		for(int j = 0; j < iteration[i]; ++j) {

			// TODO: convoluted transformation
			RGBStep(NextFrame->image[i],
					LastFrame->image[i],
					NextFrame->vmap[i],
					LastFrame->vmap[i],
					NextFrame->dIdx[i],
					NextFrame->dIdy[i],
					NextFrame->GpuRotation(),
					NextFrame->GpuInvRotation(),
					LastFrame->GpuRotation(),
					LastFrame->GpuInvRotation(),
					NextFrame->GpuTranslation(),
					LastFrame->GpuTranslation(),
					K(i),
					sumSE3,
					outSE3,
					sumRes,
					outRes,
					rgbResidual,
					matA_rgb.data(),
					vecb_rgb.data());

			rgbError = sqrt(rgbResidual[0]) / rgbResidual[1];
			rgbCount = (int) rgbResidual[1];

			if (std::isnan(rgbError) || rgbCount < minIcpCount[i]) {
					std::cout << "track rgb failed" << std::endl;
					NextFrame->pose = lastPose;
					return false;
			}

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
					matA_icp.data(),
					vecb_icp.data());

			icpError = sqrt(icpResidual[0]) / icpResidual[1];
			icpCount = (int) icpResidual[1];

			if (std::isnan(icpError) || icpCount < minIcpCount[i]) {
				std::cout << "track icp failed" << std::endl;
				NextFrame->pose = lastPose;
				return false;
			}

			float w = 1e-4;
			matA = matA_rgb * w + matA_icp;
			vecb = vecb_rgb * w + vecb_icp;

//			matA = matA_icp;
//			vecb = vecb_icp;

			result = matA.ldlt().solve(vecb);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();

			delta = dT * delta;
			nextPose = lastPose * delta.inverse();
			NextFrame->pose = nextPose;
		}
	}

	Eigen::Matrix4d p = pose.inverse() * NextFrame->pose;
	Eigen::Matrix3d r = p.topLeftCorner(3, 3);
	Eigen::Vector3d t = p.topRightCorner(3, 1);
	Eigen::Vector3d a = r.eulerAngles(0, 1, 2).array().sin();
	if ((icpError < 1e-4/* && rgbError < 0.01*/) && (a.norm() <= 0.3 && t.norm() <= 0.3)) {
		return true;
	} else {
		std::cout << "bad : " << icpError << "/" << rgbError << " " << a.norm() << " " << t.norm() << std::endl;
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
			SURF & key = map->hostKeys[i];
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
		if (matches[i][0].distance < 0.95 * matches[i][1].distance) {
			refined.push_back(matches[i][0]);
		}	else if(useGraphMatching) {
			refined.push_back(matches[i][0]);
			refined.push_back(matches[i][1]);
		}
	}

	std::cout << refined.size() << std::endl;

	if (refined.size() < 3)
		return false;

	framePoints.clear();
	refPoints.clear();

	if(useGraphMatching) {
		FilterMatching();
	} else {
		for (int i = 0; i < refined.size(); ++i) {
			framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
			refPoints.push_back(mapKeys[refined[i].trainIdx]);
		}
	}
	output = refPoints;
	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool bOK = Solver::PoseEstimate(framePoints, refPoints, outliers, delta, maxIterReloc);

	if (!bOK) {
		return false;
	}

	// try to find a closest Key Frame in the map
	// TODO: should consider topological relations
//	KeyFrame * CandidateKF = NULL;
//	float norm_rot, norm_trans;
//	std::set<const KeyFrame *>::iterator iter = map->keyFrames.begin();
//	std::set<const KeyFrame *>::iterator lend = map->keyFrames.end();
//	for(; iter != lend; ++iter) {
//		const KeyFrame * kf = *iter;
//		Eigen::Matrix4d pose = delta * kf->pose.cast<double>();
//		Eigen::Matrix3d r = pose.topLeftCorner(3, 3);
//		Eigen::Vector3d t = pose.topRightCorner(3, 1);
//		Eigen::Vector3d angle = r.eulerAngles(0, 1, 2).array().sin();
//		if(!CandidateKF) {
//			CandidateKF = const_cast<KeyFrame *>(kf);
//			norm_rot = angle.norm();
//			norm_trans = t.norm();
//			continue;
//		}
//
//		float nrot = angle.norm();
//		float ntrans = t.norm();
//		if((nrot + ntrans) < (norm_rot + norm_trans)) {
//			CandidateKF = const_cast<KeyFrame *>(kf);
//			norm_rot = angle.norm();
//			norm_trans = t.norm();
//		}
//	}

	std::cout << "Reloc Sucess." << std::endl;
	NextFrame->pose = delta.inverse();
//	ReferenceKF = CandidateKF;

	return true;
}

void Tracker::FilterMatching() {

	// used for storing key points in the current frame
	frameKeySelected.clear();
	// used for storing key points in the map (current time slice)
	mapKeySelected.clear();
	// store distances between all matches.
	keyDistance.clear();
	// frame key indices
	queryKeyIdx.clear();

	// build a list of query key points for the current frame
	for (int i = 0; i < refined.size(); ++i) {

		// get information from key matches
		int trainIdx = refined[i].trainIdx;
		int queryIdx = refined[i].queryIdx;
		SURF & trainKey = map->hostKeys[trainIdx];

		if (!trainKey.valid)
			continue;

		SURF queryKey;
		Eigen::Vector3f & p = NextFrame->mapPoints[queryIdx];
		queryKey.pos = { p(0), p(1), p(2) };
		queryKey.normal = NextFrame->pointNormal[queryIdx];
		frameKeySelected.push_back(queryKey);
		mapKeySelected.push_back(trainKey);
		keyDistance.push_back(refined[i].distance);
		queryKeyIdx.push_back(queryIdx);
	}

	DeviceArray<SURF> trainKeys(mapKeySelected);
	DeviceArray<SURF> queryKeys(frameKeySelected);
	DeviceArray<float> matchDist(keyDistance);
	DeviceArray<int> queryIdx(queryKeyIdx);

	cuda::GpuMat AdjecencyMatrix(frameKeySelected.size(), frameKeySelected.size(), CV_32FC1);

	// build adjacency matrix from raw key point matches
	DeviceArray<int> SelectedIdx;
	DeviceArray<SURF> queryFiltered, trainFiltered;
	BuildAdjecencyMatrix(AdjecencyMatrix, queryKeys, trainKeys, matchDist);

	// filtered out useful key points
	FilterKeyMatching(AdjecencyMatrix, trainKeys, queryKeys, trainFiltered, queryFiltered, queryIdx, SelectedIdx);

	std::vector<int> keyIdxFiltered_cpu;
	std::vector<SURF> trainKeyFiltered_cpu;
	std::vector<SURF> queryKeyFiltered_cpu;

	trainFiltered.download(trainKeyFiltered_cpu);
	queryFiltered.download(queryKeyFiltered_cpu);
	SelectedIdx.download(keyIdxFiltered_cpu);

	// filter out redundant feature matchings
	Eigen::Vector3d p, q;
	for (int i = 0; i < queryFiltered.size; ++i) {
		if (!queryKeyFiltered_cpu[i].valid || !trainKeyFiltered_cpu[i].valid)
			continue;

		bool redundant = false;
		for (int j = 0; j < i; j++) {
			if (keyIdxFiltered_cpu[j] == keyIdxFiltered_cpu[i]) {
				redundant = true;
				break;
			}
		}
		if (redundant)
			continue;

		p << queryKeyFiltered_cpu[i].pos.x, queryKeyFiltered_cpu[i].pos.y, queryKeyFiltered_cpu[i].pos.z;
		q << trainKeyFiltered_cpu[i].pos.x, trainKeyFiltered_cpu[i].pos.y, trainKeyFiltered_cpu[i].pos.z;
		framePoints.push_back(p);
		refPoints.push_back(q);
	}
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

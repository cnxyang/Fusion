#include "Timer.h"
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

Tracker::Tracker(int w, int h, float fx, float fy, float cx, float cy)
	:referenceKF(nullptr), lastKF(nullptr), graphMatching(false),
	 useIcp(true), useSo3(true), state(1), needImages(false),
	 lastState(1), lastReloc(0), imageUpdated(false), localisationOnly(false) {

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

	sumSE3.create(29, 96);
	outSE3.create(29);
	sumSO3.create(11, 96);
	outSO3.create(11);

	iteration[0] = 10;
	iteration[1] = 5;
	iteration[2] = 3;

	lastIcpError = std::numeric_limits<float>::max();
	lastSo3Error = std::numeric_limits<float>::max();

	K = Intrinsics(fx, fy, cx, cy);

	lastRelocId = 0;
	orbExtractor = cuda::ORB::create(1500);
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
	std::swap(state, lastState);

	if(needImages) {
		if(state != -1)
			RenderImage(lastVMap[0], lastNMap[0], make_float3(0, 0, 0), renderedImage);
		depthToImage(nextDepth[0], renderedDepth);
		rgbImageToRgba(color, rgbaImage);
		imageUpdated = true;
	}

	switch(state) {
	case 1:
		initTracking();
		swapFrame();
		lastState = 0;
		return true;

	case 0:
		valid = trackFrame(false);

		if(valid) {
			lastState = 0;
			currentPose = nextFrame.pose;
			if(state != -1) {
				if(needNewKF())
					createNewKF();
				swapFrame();
				return true;
			}
			else
				return false;
		}

		lastState = -1;
		return false;

	case -1:
		std::cout << "tracking failed" << std::endl;
		valid = relocalise();

		if(valid) {
			std::cout << "relocalisation success" << std::endl;
			lastState = 0;
			currentPose = nextFrame.pose;
			lastRelocId = nextFrame.frameId;
			swapFrame();
			return true;
		}
		else {
			lastState = -1;
			return false;
		}
	}
}

void Tracker::fuseMapPoint() {
	mpMap->fuseKeys(nextFrame, outliers);
}

bool Tracker::trackFrame(bool useKF) {

	Timer::Start("track", "trackframe");
	bool valid = false;
	if(useKF)
		valid = trackReferenceKF();
	else
		valid = trackLastFrame();
	if(!valid) {
		return false;
	}
	nextFrame.SetPose(lastFrame);
	initIcp();
	valid = computeSE3();
	Timer::Stop("track", "trackframe");

	return valid;
}

void Tracker::extractFeatures() {
	cv::cuda::GpuMat tmpImage(480, 640, CV_8UC1, nextImage[0].data, nextImage[0].step);
	orbExtractor->detectAndCompute(tmpImage, cv::cuda::GpuMat(), nextFrame.keys, nextFrame.descriptors);
}

bool Tracker::trackReferenceKF() {

	std::vector<cv::DMatch> refined;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	orbMatcher->knnMatch(nextFrame.descriptors, referenceKF->frameDescriptors, rawMatches, 2);

	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.85 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	N = refined.size();
	if (N < 3)
		return false;

	std::vector<Eigen::Vector3d> src, ref;
	for (int i = 0; i < N; ++i) {
		src.push_back(nextFrame.mPoints[refined[i].queryIdx]);
		ref.push_back(referenceKF->frameKeys[refined[i].trainIdx]);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool result = Solver::SolveAbsoluteOrientation(src, ref, outliers, delta, maxIter);

	N = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		nextPose = delta.inverse() * referenceKF->pose;
		nextFrame.pose = nextPose;
		nextFrame.deltaPose = delta.inverse();
	}

	return result;
}

bool Tracker::trackLastFrame() {

	std::vector<cv::DMatch> refined;
	std::vector<std::vector<cv::DMatch>> rawMatches;
	orbMatcher->knnMatch(nextFrame.descriptors, lastFrame.descriptors, rawMatches, 2);

	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.85 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	N = refined.size();
	if (N < 3)
		return false;

	std::vector<Eigen::Vector3d> src, ref;
	for (int i = 0; i < N; ++i) {
		src.push_back(nextFrame.mPoints[refined[i].queryIdx]);
		ref.push_back(lastFrame.mPoints[refined[i].trainIdx]);
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool result = Solver::SolveAbsoluteOrientation(src, ref, outliers, delta, maxIter);

	outliers.resize(refined.size());
	N = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		nextPose = delta.inverse() * lastFrame.pose;
		nextFrame.pose = nextPose;
		nextFrame.deltaPose = delta.inverse() * lastFrame.deltaPose;

		nextFrame.index.resize(nextFrame.N);
		fill(nextFrame.index.begin(), nextFrame.index.end(), -1);
		for(int i = 0; i < outliers.size(); ++i) {
			if(!outliers[i]) {
				nextFrame.index[refined[i].queryIdx] = lastFrame.index[refined[i].trainIdx];
			}
		}
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
		BackProjectPoints(nextDepth[i],
						  nextVMap[i],
						  Frame::mDepthCutoff,
						  Frame::fx(i),
						  Frame::fy(i),
						  Frame::cx(i),
						  Frame::cy(i));
		ComputeNormalMap(nextVMap[i], nextNMap[i]);
	}
}

void Tracker::swapFrame() {

	lastFrame = Frame(nextFrame);

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

	if(localisationOnly)
		return false;

//	if(nextFrame.frameId > 5 && nextFrame.frameId - lastRelocId < 5)
//		return false;
//
//	if(N < 100)
//		return true;

	if(rotationChanged() >= 0.2 || translationChanged() >= 0.2)
		return true;

	return false;
}

void Tracker::createNewKF() {

	std::swap(lastKF, referenceKF);
	if(lastKF)
		lastKF->frameDescriptors.release();
	referenceKF = new KeyFrame(&nextFrame);
	nextFrame.deltaPose = Eigen::Matrix4d::Identity();
	mpMap->push_back(referenceKF);
	nextFrame.index = referenceKF->keyIndices;
}

void Tracker::computeSO3() {

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> matrixA;
	Eigen::Matrix<double, 3, 1> vectorB;
	Eigen::Matrix<double, 3, 1> result;
	lastSo3Error = std::numeric_limits<float>::max();
	Eigen::Matrix3d lastRot = lastFrame.Rotation();
	Eigen::Matrix3d nextRot = nextFrame.Rotation();
	Eigen::Matrix3d final = lastRot.inverse() * nextRot;

	for(int i = NUM_PYRS - 1; i >= 0; --i) {

		Eigen::Matrix3d matK = Eigen::Matrix3d::Identity();
		matK(0, 0) = K(i).fx;
		matK(1, 1) = K(i).fy;
		matK(0, 2) = K(i).cx;
		matK(1, 2) = K(i).cy;

		for(int j = 0; j < 10; ++j) {

			Eigen::Matrix3d homography = matK * final * matK.inverse();
			Eigen::Matrix3d Kinv = matK.inverse();
			Eigen::Matrix3d KRlr = matK * final;
			SO3Step(nextImage[i],
					lastImage[i],
					eigen_to_mat3f(homography),
					eigen_to_mat3f(Kinv),
					eigen_to_mat3f(KRlr),
					sumSO3,
					outSO3,
					so3Residual,
					matrixA.data(),
					vectorB.data());

			float so3Error = sqrt(so3Residual[0]) / so3Residual[1];
			int so3Count = (int) so3Residual[1];
			std::cout << so3Error << std::endl;

			result = matrixA.ldlt().solve(vectorB);
			auto e = Sophus::SO3d::exp(result);
			auto dT = e.matrix();
			std::cout << dT << std::endl;
			final = dT * final;
		}
	}
	nextFrame.pose.topLeftCorner(3, 3) = final * lastRot;
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

			ICPStep(nextVMap[i],
					lastVMap[i],
					nextNMap[i],
					lastNMap[i],
					nextFrame.Rot_gpu(),
					nextFrame.Trans_gpu(),
					lastFrame.Rot_gpu(),
					lastFrame.RotInv_gpu(),
					lastFrame.Trans_gpu(),
					K(i),
					sumSE3,
					outSE3,
					icpResidual,
					matA.data(),
					vecb.data());

			float icpError = sqrt(icpResidual[0]) / icpResidual[1];
			int icpCount = (int) icpResidual[1];

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

	if(lastState != -1) {
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
		else if (graphMatching) {
			matches.push_back(rawMatches[i][0]);
			matches.push_back(rawMatches[i][1]);
		}
	}

	if (matches.size() < 50)
		return false;

	std::vector<Eigen::Vector3d> plist;
	std::vector<Eigen::Vector3d> qlist;
	if (graphMatching) {
		std::vector<ORBKey> vFrameKey;
		std::vector<ORBKey> vMapKey;
		std::vector<float> vDistance;
		std::vector<int> vQueryIdx;
		cv::Mat cpuFrameDesc;
		nextFrame.descriptors.download(cpuFrameDesc);
		cv::Mat cpuMatching(2, matches.size(), CV_32SC1);

		for (int i = 0; i < matches.size(); ++i) {
			int trainIdx = matches[i].trainIdx;
			int queryIdx = matches[i].queryIdx;
			ORBKey trainKey = mpMap->hostKeys[trainIdx];
			ORBKey queryKey;
			if (trainKey.valid && queryKey.valid) {
				cv::Vec3f normal = nextFrame.mNormals[queryIdx];
				Eigen::Vector3d& p = nextFrame.mPoints[queryIdx];
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

		MatchDist.upload((void*) vDistance.data(), vDistance.size());
		trainKeys.upload((void*) vMapKey.data(), vMapKey.size());
		queryKeys.upload((void*) vFrameKey.data(), vFrameKey.size());
		QueryIdx.upload((void*) vQueryIdx.data(), vQueryIdx.size());

		cuda::GpuMat AdjecencyMatrix(matches.size(), matches.size(), CV_32FC1);
		DeviceArray<ORBKey> query_select, train_select;
		DeviceArray<int> SelectedIdx;

		BuildAdjecencyMatrix(AdjecencyMatrix, trainKeys, queryKeys, MatchDist,
				train_select, query_select, QueryIdx, SelectedIdx);

		std::vector<int> vSelectedIdx;
		std::vector<ORBKey> vORB_train, vORB_query;
		vSelectedIdx.resize(SelectedIdx.size);
		vORB_train.resize(train_select.size);
		vORB_query.resize(query_select.size);

		train_select.download((void*) vORB_train.data(), vORB_train.size());
		query_select.download((void*) vORB_query.data(), vORB_query.size());
		SelectedIdx.download((void*) vSelectedIdx.data(), vSelectedIdx.size());

		for (int i = 0; i < query_select.size; ++i) {
			Eigen::Vector3d p, q;
			if (vORB_query[i].valid && vORB_train[i].valid) {
				bool redundant = false;
				for (int j = 0; j < i; j++) {
					if (vSelectedIdx[j] == vSelectedIdx[i]) {
						redundant = true;
						break;
					}
				}
				if (!redundant) {
					p << vORB_query[i].pos.x, vORB_query[i].pos.y, vORB_query[i].pos.z;
					q << vORB_train[i].pos.x, vORB_train[i].pos.y, vORB_train[i].pos.z;
					plist.push_back(p);
					qlist.push_back(q);
				}
			}
		}
	}
	else {
		for (int i = 0; i < matches.size(); ++i) {
			plist.push_back(nextFrame.mPoints[matches[i].queryIdx]);
			qlist.push_back(mapPoints[matches[i].trainIdx]);
		}
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool bOK = Solver::SolveAbsoluteOrientation(plist, qlist, outliers, delta, maxIterReloc);

	if (!bOK) {
		return false;
	}

	nextFrame.index.resize(nextFrame.N);
	fill(nextFrame.index.begin(), nextFrame.index.end(), -1);
	for(int i = 0; i < matches.size(); ++i) {
		if(!outliers[i]) {
			nextFrame.index[matches[i].queryIdx] = mpMap->hostIndex[matches[i].trainIdx];
		}
	}
	nextFrame.SetPose(delta.inverse());
	return true;
}

Eigen::Matrix4f Tracker::getCurrentPose() const {
	return currentPose.cast<float>();
}

void Tracker::setMap(Mapping* pMap) {
	mpMap = pMap;
}

void Tracker::setViewer(Viewer* pViewer) {
	mpViewer = pViewer;
}

#include "Frame.hpp"
#include "cufunc.h"
#include "Timer.hpp"
#include <Eigen/Dense>

using namespace cv;
using namespace std;

Mat Frame::mK[NUM_PYRS];
bool Frame::mbFirstCall = true;
float Frame::mDepthCutoff = 8.0f;
float Frame::mDepthScale = 1000.0f;
int Frame::mCols[NUM_PYRS];
int Frame::mRows[NUM_PYRS];
unsigned long Frame::nextId = 0;
Ptr<cuda::ORB> Frame::mORB;

Frame::Frame():N(0) {}

Frame::Frame(const Frame& other):N(0) {

	mPoints = other.mPoints;
	keys = other.keys;
	other.descriptors.copyTo(descriptors);

	pose = other.pose;
//	mPoseInv = other.mPoseInv;
}

bool computeVertexAndNormal(const cv::Mat & imD, float & x, float & y,
		float3 v00, float3 & n, float invfx, float invfy, float cx, float cy,
		int cols, int rows, float depthScale, float depthCutoff) {

	if(std::isnan(v00.x))
		return false;

	float3 v01;
	int x01 = (int) (x + 1.5);
	int y01 = (int) (y + 0.5);
	if(x01 < 0 || x01 >= cols || y01 < 0 || y01 >= rows)
		return false;

	v01.z = imD.at<unsigned short>(y01, x01) / depthScale;
	if(std::isnan(v01.z) || v01.z > depthCutoff)
		return false;

	v01.x = v01.z * (x01 - cx) * invfx;
	v01.y = v01.z * (y01 - cy) * invfy;

	float3 v10;
	int x10 = (int) (x + 0.5);
	int y10 = (int) (y + 1.5);
	if(x10 < 0 || x10 >= cols || y10 < 0 || y10 >= rows)
		return false;

	v10.z = imD.at<unsigned short>(y10, x10) / depthScale;
	if(std::isnan(v10.z) || v10.z > depthCutoff)
		return false;

	v10.x = v10.z * (x10 - cx) * invfx;
	v10.y = v10.z * (y10 - cy) * invfy;

	n = normalised(cross(v01 - v00, v10 - v00));

	if(std::isnan(n.x))
		return false;

	return true;
}

Frame::Frame(const DeviceArray2D<uchar> & img, const cv::Mat & imD, KeyFrame * kf) {

	if(mbFirstCall) {
		mORB = cv::cuda::ORB::create(1000);
		for(int i = 0; i < NUM_PYRS; ++i) {
			mCols[i] = imD.cols / (1 << i);
			mRows[i] = imD.rows / (1 << i);
		}
		mbFirstCall = false;
	}

	frameId = nextId++;

	rawDepth = imD;
	referenceKF = kf;

	cv::cuda::GpuMat cudaImage(img.rows(), img.cols(), CV_8UC1, img.data(), img.step());

	cv::Mat desc, descTemp;
	cuda::GpuMat DescTemp;
	std::vector<KeyPoint> KPTemp;

	Timer::Start("test", "create frame");
	mORB->detectAndCompute(cudaImage, cuda::GpuMat(), KPTemp, DescTemp);
	Timer::Stop("test", "create frame");
	N = KPTemp.size();
	if (N <= 0)
		return;

	float invfx = 1.0 / fx(0);
	float invfy = 1.0 / fy(0);
	float cx0 = cx(0);
	float cy0 = cy(0);
	//		cv::Mat cpuNormal(rows(0), cols(0), CV_32FC3);
	DescTemp.download(descTemp);

	for (int i = 0; i < N; ++i) {
		cv::KeyPoint& kp = KPTemp[i];
		float x = kp.pt.x;
		float y = kp.pt.y;
		float dp = (float) imD.at<unsigned short>((int) (y + 0.5),
				(int) (x + 0.5)) / mDepthScale;
		Eigen::Vector3d pos = Eigen::Vector3d::Zero();
		if (dp > 1e-1 && dp < mDepthCutoff) {
			pos(2) = dp;
			pos(0) = dp * (x - cx0) * invfx;
			pos(1) = dp * (y - cy0) * invfy;
			float3 normal;
			float3 v00 = { (float) pos(0), (float) pos(1), (float) pos(2) };

			bool valid = computeVertexAndNormal(imD, x, y, v00, normal, invfx,
					invfy, cx0, cy0, cols(0), rows(0), mDepthScale,
					mDepthCutoff);
			if (!valid)
				continue;

			cv::Vec3f n;
			n(0) = normal.x;
			n(1) = normal.y;
			n(2) = normal.z;
			mPoints.push_back(pos);
			mNormals.push_back(n);
			keys.push_back(kp);
			desc.push_back(descTemp.row(i));
		}
	}
	N = keys.size();
	descriptors.upload(desc);

	SetPose(Eigen::Matrix4d::Identity());
}

Frame::Frame(const cv::Mat& imRGB, const cv::Mat& imD) {

	SetPose(Eigen::Matrix4d::Identity());

	if(mbFirstCall) {
		mORB = cv::cuda::ORB::create(1000);
		for(int i = 0; i < NUM_PYRS; ++i) {
			mCols[i] = imD.cols / (1 << i);
			mRows[i] = imD.rows / (1 << i);
		}
		mbFirstCall = false;
	}

	rawDepth = imD;
	rawColor = imRGB;
	cv::Mat img;
	cv::cvtColor(imRGB, img, cv::COLOR_BGR2GRAY);
	cv::cuda::GpuMat Image(img);

	cv::Mat desc, descTemp;
	cuda::GpuMat DescTemp;
	std::vector<KeyPoint> KPTemp;

	mORB->detectAndCompute(Image, cuda::GpuMat(), KPTemp, DescTemp);
	N = KPTemp.size();
	if (N <= 0)
		return;

	float invfx = 1.0 / fx(0);
	float invfy = 1.0 / fy(0);
	float cx0 = cx(0);
	float cy0 = cy(0);
//		cv::Mat cpuNormal(rows(0), cols(0), CV_32FC3);
	DescTemp.download(descTemp);

	Timer::Start("test", "keypoint");
	for (int i = 0; i < N; ++i) {
		cv::KeyPoint& kp = KPTemp[i];
		float x = kp.pt.x;
		float y = kp.pt.y;
		float dp = (float) imD.at<unsigned short>((int) (y + 0.5),
				(int) (x + 0.5)) / mDepthScale;
		Eigen::Vector3d pos = Eigen::Vector3d::Zero();
		if (dp > 1e-1 && dp < mDepthCutoff) {
			pos(2) = dp;
			pos(0) = dp * (x - cx0) * invfx;
			pos(1) = dp * (y - cy0) * invfy;
			float3 normal;
			float3 v00 = { (float) pos(0), (float) pos(1), (float) pos(2) };

			bool valid = computeVertexAndNormal(imD, x, y, v00, normal, invfx,
					invfy, cx0, cy0, cols(0), rows(0), mDepthScale,
					mDepthCutoff);
			if (!valid)
				continue;

			cv::Vec3f n;
			n(0) = normal.x;
			n(1) = normal.y;
			n(2) = normal.z;
			mPoints.push_back(pos);
			mNormals.push_back(n);
			keys.push_back(kp);
			desc.push_back(descTemp.row(i));
		}
	}
	N = keys.size();
	descriptors.upload(desc);
	Timer::Stop("test", "keypoint");

}

void Frame::SetPose(const Frame& frame) {
	pose = frame.pose;
//	mPoseInv = frame.mPoseInv;
}

Matrix3f Frame::Rot_gpu() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(pose(0, 0), pose(0, 1), pose(0, 2));
	Rot.rowy = make_float3(pose(1, 0), pose(1, 1), pose(1, 2));
	Rot.rowz = make_float3(pose(2, 0), pose(2, 1), pose(2, 2));
	return Rot;
}

Matrix3f Frame::RotInv_gpu() const {
	Matrix3f Rot;
	const Eigen::Matrix3d mPoseInv = Rotation().transpose();
	Rot.rowx = make_float3(mPoseInv(0, 0), mPoseInv(0, 1), mPoseInv(0, 2));
	Rot.rowy = make_float3(mPoseInv(1, 0), mPoseInv(1, 1), mPoseInv(1, 2));
	Rot.rowz = make_float3(mPoseInv(2, 0), mPoseInv(2, 1), mPoseInv(2, 2));
	return Rot;
}

float3 Frame::Trans_gpu() const {
	return make_float3(pose(0, 3), pose(1, 3), pose(2, 3));
}

void Frame::SetPose(const Eigen::Matrix4d T) {
	pose = T;
}

Eigen::Matrix3d Frame::Rotation() const {
	return pose.topLeftCorner(3, 3);
}

Eigen::Vector3d Frame::Translation() {
	return pose.topRightCorner(3, 1);
}

void Frame::SetK(cv::Mat& K) {
	for(int i = 0; i < NUM_PYRS; ++i) {
		mK[i] = cv::Mat::eye(3, 3, CV_32FC1);
		mK[i].at<float>(0, 0) = K.at<float>(0, 0) / (1 << i);
		mK[i].at<float>(1, 1) = K.at<float>(1, 1) / (1 << i);
		mK[i].at<float>(0, 2) = K.at<float>(0, 2) / (1 << i);
		mK[i].at<float>(1, 2) = K.at<float>(1, 2) / (1 << i);
	}
}

float Frame::fx(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(0, 0);
}

float Frame::fy(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(1, 1);
}

float Frame::cx(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(0, 2);
}

float Frame::cy(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(1, 2);
}

int Frame::cols(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mCols[pyr];
}

int Frame::rows(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mRows[pyr];
}

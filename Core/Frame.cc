#include "Frame.h"
#include "KeyFrame.h"
#include "Reduction.h"

#include <Eigen/Dense>
#include <core/cuda.hpp>

using namespace cv;
using namespace std;

Mat Frame::mK[NUM_PYRS];
bool Frame::mbFirstCall = true;
float Frame::mDepthCutoff = 3.0f;
float Frame::mDepthScale = 1000.0f;
int Frame::mCols[NUM_PYRS];
int Frame::mRows[NUM_PYRS];
unsigned long Frame::nextId = 0;
cv::cuda::SURF_CUDA Frame::surfExt;
cv::Ptr<cv::BRISK> Frame::briskExt;

Frame::Frame():N(0) {}

void Frame::Create(int cols_, int rows_) {

	if(mbFirstCall) {
		surfExt = cv::cuda::SURF_CUDA(10, 4, 2, false, 0.005);
		briskExt = cv::BRISK::create(30, 4);
		for(int i = 0; i < NUM_PYRS; ++i) {
			mCols[i] = cols_ / (1 << i);
			mRows[i] = rows_ / (1 << i);
		}
		mbFirstCall = false;
	}

	range.create(cols_, rows_);
	color.create(cols_, rows_);

	for(int i = 0; i < NUM_PYRS; ++i) {
		int cols = cols_ / (1 << i);
		int rows = rows_ / (1 << i);
		vmap[i].create(cols, rows);
		nmap[i].create(cols, rows);
		depth[i].create(cols, rows);
		image[i].create(cols, rows);
	}
}

void Frame::FillImages(const cv::Mat & range_, const cv::Mat & color_) {

	range.upload(range_.data, range_.step);
	color.upload(color_.data, color_.step);
	FilterDepth(range, depth[0], mDepthScale);
	ImageToIntensity(color, image[0]);
	for(int i = 1; i < NUM_PYRS; ++i) {
		PyrDownGauss(depth[i - 1], depth[i]);
		PyrDownGauss(image[i - 1], image[i]);
	}

	for(int i = 0; i < NUM_PYRS; ++i) {
		ComputeVMap(depth[i], vmap[i], fx(i), fy(i), cx(i), cy(i), mDepthCutoff);
		ComputeNMap(vmap[i], nmap[i]);
	}
}

void Frame::ResizeImages() {
	for(int i = 1; i < NUM_PYRS; ++i) {
		ResizeMap(vmap[i - 1], nmap[i - 1], vmap[i], nmap[i]);
	}
}

void Frame::ClearKeyPoints() {
	N = 0;
	keys.clear();
	pt3d.clear();
	descriptors.release();
}

float Frame::InterpDepth(cv::Mat & map, float & x, float & y) {

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));

	float d00 = map.at<float>(upperLeft.y, upperLeft.x);
	float d10 = map.at<float>(lowerLeft.y, lowerLeft.x);
	float d01 = map.at<float>(upperRight.y, upperRight.x);
	float d11 = map.at<float>(lowerRight.y, lowerRight.x);

	float d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	return (1 - coeff.y) * d0 + coeff.y * d1;
}

float4 Frame::InterpNormal(cv::Mat & map, float & x, float & y) {

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));

	cv::Vec4f n = map.at<cv::Vec4f>(upperLeft.y, upperLeft.x);
	float4 d00 = make_float4(n(0), n(1), n(2), n(3));
	n = map.at<cv::Vec4f>(lowerLeft.y, lowerLeft.x);
	float4 d10 = make_float4(n(0), n(1), n(2), n(3));
	n = map.at<cv::Vec4f>(upperRight.y, upperRight.x);
	float4 d01 = make_float4(n(0), n(1), n(2), n(3));
	n = map.at<cv::Vec4f>(lowerRight.y, lowerRight.x);
	float4 d11 = make_float4(n(0), n(1), n(2), n(3));

	float4 d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float4 d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	return d0 * (1 - coeff.y) + d1 * coeff.y;
}

void Frame::ExtractKeyPoints() {

	cv::Mat rawDescriptors;
	cv::Mat sNormal(depth[0].rows, depth[0].cols, CV_32FC4);
	cv::Mat sDepth(depth[0].rows, depth[0].cols, CV_32FC1);
	std::vector<cv::KeyPoint> rawKeyPoints;

	depth[0].download(sDepth.data, sDepth.step);
	nmap[0].download(sNormal.data, sNormal.step);

	cv::cuda::GpuMat img(image[0].rows, image[0].cols, CV_8UC1, image[0].data, image[0].step);
	surfExt(img, cv::cuda::GpuMat(), rawKeyPoints, descriptors);
	descriptors.download(rawDescriptors);

	cv::Mat desc;

	float invfx = 1.0 / fx(0);
	float invfy = 1.0 / fy(0);
	float cx0 = cx(0);
	float cy0 = cy(0);

	for(int i = 0; i < rawKeyPoints.size(); ++i) {
		cv::KeyPoint & kp = rawKeyPoints[i];
		float & x = kp.pt.x;
		float & y = kp.pt.y;
		float dp = InterpDepth(sDepth, x, y);
		if(!std::isnan(dp) && dp > 1e-3 && dp < mDepthCutoff) {
			float4 n = InterpNormal(sNormal, x, y);
			if(!std::isnan(n.x)) {
				float3 v;
				v.z = dp;
				v.x = dp * (x - cx0) * invfx;
				v.y = dp * (y - cy0) * invfy;

				pt3d.push_back(v);
				normal.push_back(n);
				keys.push_back(kp);
				desc.push_back(rawDescriptors.row(i));
			}
		}
	}

	N = pt3d.size();
	descriptors.upload(desc);
	SetPose(Eigen::Matrix4d::Identity());

//	cv::Mat rawImage(480, 640, CV_8UC3);
//	color.download(rawImage.data, rawImage.step);
//	for(int i = 0; i < N; ++i) {
//		cv::Point2f upperLeft = keys[i].pt - cv::Point2f(5, 5);
//		cv::Point2f lowerRight = keys[i].pt + cv::Point2f(5, 5);
//		cv::drawMarker(rawImage, keys[i].pt, cv::Scalar(0, 125, 0), cv::MARKER_CROSS, 5);
//		cv::rectangle(rawImage, upperLeft, lowerRight, cv::Scalar(0, 125, 0));
//	}
//
//	cv::imshow("img", rawImage);
//	cv::waitKey(10);
}

void Frame::SetPose(const Frame& frame) {
	pose = frame.pose;
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

// refactorying

void Frame::setPose(Frame * other) {
	pose = other->pose;
}

void Frame::setPose(Eigen::Matrix4d & newPose) {
	pose = newPose;
}

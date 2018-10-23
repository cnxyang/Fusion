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

Frame::Frame():frameId(0), N(0), bad(false) {}

Frame::Frame(const Frame * other):frameId(other->frameId), N(0), bad(false) {

	matRange.create(Frame::rows(0), Frame::cols(0), CV_32FC1);
	matColor.create(Frame::rows(0), Frame::cols(0), CV_8UC3);
	matNormal.create(Frame::rows(0), Frame::cols(0), CV_32FC4);
	other->range.download(matRange.data, matRange.step);
	other->color.download(matColor.data, matColor.step);
	other->nmap[0].download(matNormal.data, matNormal.step);
}

void Frame::operator=(const Frame & other) {
	matRange = other.matRange;
	matColor = other.matColor;
	matNormal = other.matNormal;
	frameId = other.frameId;
}

void Frame::Create(int cols_, int rows_) {

	if(mbFirstCall) {
		surfExt = cv::cuda::SURF_CUDA(20);
		briskExt = cv::BRISK::create(30, 4);
		for(int i = 0; i < NUM_PYRS; ++i) {
			mCols[i] = cols_ / (1 << i);
			mRows[i] = rows_ / (1 << i);
		}
		mbFirstCall = false;
	}

	temp.create(cols_, rows_);
	range.create(cols_, rows_);
	color.create(cols_, rows_);

	for(int i = 0; i < NUM_PYRS; ++i) {
		int cols = cols_ / (1 << i);
		int rows = rows_ / (1 << i);
		vmap[i].create(cols, rows);
		nmap[i].create(cols, rows);
		depth[i].create(cols, rows);
		image[i].create(cols, rows);
		dIdx[i].create(cols, rows);
		dIdy[i].create(cols, rows);
	}
}

void Frame::Clear() {

}

void Frame::FillImages(const cv::Mat & range_, const cv::Mat & color_) {

	temp.upload(range_.data, range_.step);
	color.upload(color_.data, color_.step);
	FilterDepth(temp, range, depth[0], mDepthScale, mDepthCutoff);
	ImageToIntensity(color, image[0]);
	for(int i = 1; i < NUM_PYRS; ++i) {
		PyrDownGauss(depth[i - 1], depth[i]);
		PyrDownGauss(image[i - 1], image[i]);
	}

	for(int i = 0; i < NUM_PYRS; ++i) {
		ComputeVMap(depth[i], vmap[i], fx(i), fy(i), cx(i), cy(i), mDepthCutoff);
		ComputeNMap(vmap[i], nmap[i]);
		ComputeDerivativeImage(image[i], dIdx[i], dIdy[i]);
	}

	frameId = nextId++;
	bad = false;
}

void Frame::ResizeImages() {
	for(int i = 1; i < NUM_PYRS; ++i) {
		ResizeMap(vmap[i - 1], nmap[i - 1], vmap[i], nmap[i]);
	}
}

void Frame::ClearKeyPoints() {
	N = 0;
	keyPoints.clear();
	mapPoints.clear();
	descriptors.release();
}

float Frame::InterpDepth(cv::Mat & map, float & x, float & y) {

	float dp = std::nanf("0x7fffffff");
	if(x <= 1 || y <= 1 || y >= map.cols - 1 || x >= map.rows - 1)
		return dp;

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));

	float d00 = map.at<float>(upperLeft.y, upperLeft.x);
	if(std::isnan(d00) || d00 < 0.3 || d00 > mDepthCutoff)
		return dp;

	float d10 = map.at<float>(lowerLeft.y, lowerLeft.x);
	if(std::isnan(d10) || d10 < 0.3 || d10 > mDepthCutoff)
		return dp;

	float d01 = map.at<float>(upperRight.y, upperRight.x);
	if(std::isnan(d01) || d01 < 0.3 || d01 > mDepthCutoff)
		return dp;

	float d11 = map.at<float>(lowerRight.y, lowerRight.x);
	if(std::isnan(d11) || d11 < 0.3 || d11 > mDepthCutoff)
		return dp;

	float d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	float final = (1 - coeff.y) * d0 + coeff.y * d1;
	if(std::abs(final - d00) <= 0.005)
		dp = final;
	return dp;
}

float4 Frame::InterpNormal(cv::Mat & map, float & x, float & y) {

	if(x <= 1 || y <= 1 || y >= map.cols - 1 || x >= map.rows - 1)
		return make_float4(std::nanf("0x7fffffff"));

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));
	cv::Vec4f n;

	n = map.at<cv::Vec4f>(upperLeft.y, upperLeft.x);
	float4 d00 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(lowerLeft.y, lowerLeft.x);
	float4 d10 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(upperRight.y, upperRight.x);
	float4 d01 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(lowerRight.y, lowerRight.x);
	float4 d11 = make_float4(n(0), n(1), n(2), n(3));

	float4 d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float4 d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	float4 final = d0 * (1 - coeff.y) + d1 * coeff.y;

	if(norm(final - d00) <= 0.1)
		return final;
	else
		return make_float4(std::nanf("0x7fffffff"));
}

void Frame::ExtractKeyPoints() {

	cv::Mat rawDescriptors;
	cv::Mat sNormal(depth[0].rows, depth[0].cols, CV_32FC4);
	cv::Mat sDepth(depth[0].rows, depth[0].cols, CV_32FC1);
	std::vector<cv::KeyPoint> rawKeyPoints;

	depth[0].download(sDepth.data, sDepth.step);
	nmap[0].download(sNormal.data, sNormal.step);

	N = 0;
	keyPoints.clear();
	mapPoints.clear();
	descriptors.release();

	cv::cuda::GpuMat img(image[0].rows, image[0].cols, CV_8UC1, image[0].data, image[0].step);
	surfExt(img, cv::cuda::GpuMat(), rawKeyPoints, descriptors);
	descriptors.download(rawDescriptors);

	cv::Mat desc;
	N = rawKeyPoints.size();
	for(int i = 0; i < N; ++i) {
		cv::KeyPoint & kp = rawKeyPoints[i];
		float x = kp.pt.x;
		float y = kp.pt.y;
		float dp = InterpDepth(sDepth, x, y);
		if(!std::isnan(dp) && dp > 0.3 && dp < mDepthCutoff) {
			float4 n = InterpNormal(sNormal, x, y);
			if(!std::isnan(n.x)) {
				Eigen::Vector3f v;
				v(0) = dp * (x - cx(0)) / fx(0);
				v(1) = dp * (y - cy(0)) / fy(0);
				v(2) = dp;
				mapPoints.push_back(v);
				keyPoints.push_back(kp);
				pointNormal.push_back(n);
				desc.push_back(rawDescriptors.row(i));
			}
		}
	}

	N = mapPoints.size();
	if(N < MIN_KEY_POINTS)
		bad = true;

	descriptors.upload(desc);
	pose = Eigen::Matrix4d::Identity();
}

void Frame::DrawKeyPoints() {

	cv::Mat rawImage(480, 640, CV_8UC1);
	image[0].download(rawImage.data, rawImage.step);
	for (int i = 0; i < N; ++i) {
		cv::Point2f upperLeft = keyPoints[i].pt - cv::Point2f(5, 5);
		cv::Point2f lowerRight = keyPoints[i].pt + cv::Point2f(5, 5);
		cv::drawMarker(rawImage, keyPoints[i].pt, cv::Scalar(0, 125, 0), cv::MARKER_CROSS, 5);
		cv::rectangle(rawImage, upperLeft, lowerRight, cv::Scalar(0, 125, 0));
	}

	cv::imshow("img", rawImage);
	cv::waitKey(10);
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

Eigen::Vector3f Frame::GetWorldPoint(int i) const {
	Eigen::Matrix3f r = Rotation().cast<float>();
	Eigen::Vector3f t = Translation().cast<float>();
	return r * mapPoints[i] + t;
}

Matrix3f Frame::GpuRotation() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(pose(0, 0), pose(0, 1), pose(0, 2));
	Rot.rowy = make_float3(pose(1, 0), pose(1, 1), pose(1, 2));
	Rot.rowz = make_float3(pose(2, 0), pose(2, 1), pose(2, 2));
	return Rot;
}

Matrix3f Frame::GpuInvRotation() const {
	Matrix3f Rot;
	const Eigen::Matrix3d mPoseInv = Rotation().transpose();
	Rot.rowx = make_float3(mPoseInv(0, 0), mPoseInv(0, 1), mPoseInv(0, 2));
	Rot.rowy = make_float3(mPoseInv(1, 0), mPoseInv(1, 1), mPoseInv(1, 2));
	Rot.rowz = make_float3(mPoseInv(2, 0), mPoseInv(2, 1), mPoseInv(2, 2));
	return Rot;
}

float3 Frame::GpuTranslation() const {
	return make_float3(pose(0, 3), pose(1, 3), pose(2, 3));
}

Eigen::Matrix3d Frame::Rotation() const {
	return pose.topLeftCorner(3, 3);
}

Eigen::Matrix3d Frame::RotationInv() const {
	return Rotation().transpose();
}

Eigen::Vector3d Frame::Translation() const {
	return pose.topRightCorner(3, 1);
}

Eigen::Vector3d Frame::TranslationInv() const {
	return -RotationInv() * Translation();
}

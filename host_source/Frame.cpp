#include "Frame.hpp"
#include "device_function.hpp"

#include <Eigen/Dense>

using namespace cv;
using namespace std;

int Frame::N[numPyrs];
Mat Frame::mK[numPyrs];
bool Frame::mbFirstCall = true;
float Frame::mDepthCutoff = 8.0f;
float Frame::mDepthScale = 1000.0f;
int Frame::mCols[numPyrs];
int Frame::mRows[numPyrs];
Ptr<cuda::ORB> Frame::mORB;

Frame::Frame():mNkp(0) {}

Frame::~Frame() { release(); }

Frame::Frame(const Frame& other):mNkp(0) {

	for(int i = 0; i < numPyrs; ++i) {
		other.mDepth[i].copyTo(mDepth[i]);
		other.mGray[i].copyTo(mGray[i]);
		other.mVMap[i].copyTo(mVMap[i]);
		other.mNMap[i].copyTo(mNMap[i]);
		other.mdIx[i].copyTo(mdIx[i]);
		other.mdIy[i].copyTo(mdIy[i]);
	}

	mPoints = other.mPoints;
	mKeyPoints = other.mKeyPoints;
	other.mDescriptors.copyTo(mDescriptors);

	mPose = other.mPose;
	mPoseInv = other.mPoseInv;
}

Frame::Frame(const Rendering& observation, Eigen::Matrix4d& pose) {

	for(int i = 0; i < numPyrs; ++i) {
		if(i == 0) {
			observation.VMap.copyTo(mVMap[0]);
			observation.NMap.copyTo(mNMap[0]);
		}
		else {
			ResizeMap(mVMap[i - 1], mNMap[i - 1], mVMap[i], mNMap[i]);
		}

		mDepth[i].create(Frame::cols(i), Frame::rows(i));
		ProjectToDepth(mVMap[i], mDepth[i]);
	}

	SetPose(pose);
}

Frame::Frame(const Frame& other, const Rendering& observation) {

	for(int i = 0; i < numPyrs; ++i) {
		if(i == 0) {
			observation.VMap.copyTo(mVMap[0]);
			observation.NMap.copyTo(mNMap[0]);
		}
		else {
			ResizeMap(mVMap[i - 1], mNMap[i - 1], mVMap[i], mNMap[i]);
		}

		mDepth[i].create(Frame::cols(i), Frame::rows(i));
		ProjectToDepth(mVMap[i], mDepth[i]);
		other.mGray[i].copyTo(mGray[i]);
		other.mdIx[i].copyTo(mdIx[i]);
		other.mdIy[i].copyTo(mdIy[i]);
	}

	mPoints = other.mPoints;
	mKeyPoints = other.mKeyPoints;
	other.mDescriptors.copyTo(mDescriptors);
	mPose = other.mPose;
	mPoseInv = other.mPoseInv;
}

Frame::Frame(const cv::Mat& imRGB, const cv::Mat& imD) {

	if(mbFirstCall) {
		mORB = cv::cuda::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::cuda::ORB::FAST_SCORE);
		for(int i = 0; i < numPyrs; ++i) {
			mCols[i] = imD.cols / (1 << i);
			mRows[i] = imD.rows / (1 << i);
			N[i] = mCols[i] * mRows[i];
		}
		mbFirstCall = false;
	}

	DeviceArray2D<uchar3> rawRGB(cols(0), rows(0));
	DeviceArray2D<ushort> rawDepth(cols(0), rows(0));
	rawRGB.upload((void*)imRGB.data, imRGB.step, cols(0), rows(0));
	rawDepth.upload((void*)imD.data, imD.step, cols(0), rows(0));
	for(int i = 0; i < numPyrs; ++i) {
		mdIx[i].create(cols(i), rows(i));
		mdIy[i].create(cols(i), rows(i));
		mGray[i].create(cols(i), rows(i));
		mVMap[i].create(cols(i), rows(i));
		mNMap[i].create(cols(i), rows(i));
		mDepth[i].create(cols(i), rows(i));
		if(i == 0) {
			BilateralFiltering(rawDepth, mDepth[0], mDepthScale);
			ColourImageToIntensity(rawRGB, mGray[0]);
		}
		else {
			PyrDownGaussian(mGray[i - 1], mGray[i]);
			PyrDownGaussian(mDepth[i - 1], mDepth[i]);
		}
		BackProjectPoints(mDepth[i], mVMap[i], mDepthCutoff, fx(i), fy(i), cx(i), cy(i));
		ComputeNormalMap(mVMap[i], mNMap[i]);
		ComputeDerivativeImage(mGray[i], mdIx[i], mdIy[i]);
	}

	cv::Mat desc, descTemp;
	cuda::GpuMat GrayTemp, DescTemp;
	vector<KeyPoint> KPTemp;
	GrayTemp.create(rows(0), cols(0), CV_8UC1);
	SafeCall(cudaMemcpy2D((void*)GrayTemp.data, GrayTemp.step,
			(void*)mGray[0], mGray[0].step(), sizeof(char) * mGray[0].cols(),
			 mGray[0].rows(), cudaMemcpyDeviceToDevice));
	mORB->detectAndCompute(GrayTemp, cuda::GpuMat(), KPTemp, DescTemp);

	mNkp = KPTemp.size();
	if (mNkp > 0) {
		float invfx = 1.0 / fx(0);
		float invfy = 1.0 / fy(0);
		float cx0 = cx(0);
		float cy0 = cy(0);
		DescTemp.download(descTemp);
		for(int i = 0; i < mNkp; ++i) {
			cv::KeyPoint& kp = KPTemp[i];
			float x = kp.pt.x;
			float y = kp.pt.y;
			float dp = (float)imD.at<unsigned short>((int)(y + 0.5), (int)(x + 0.5)) / mDepthScale;

			Eigen::Vector3d pos = Eigen::Vector3d::Zero();
			if (dp > 1e-1 && dp < mDepthCutoff) {

				pos(2) = dp;
				pos(0) = dp * (x - cx0) * invfx;
				pos(1) = dp * (y - cy0) * invfy;

				mPoints.push_back(pos);
				mKeyPoints.push_back(kp);
				desc.push_back(descTemp.row(i));
			}
		}
		mNkp = mKeyPoints.size();
		mDescriptors.upload(desc);
	}

//	CudaImage img;
//	img.Allocate(cols(0), rows(0), iAlignUp(cols(0), 128), false, nullptr, (float*)GrayImg.data);
//	InitSiftData(mSiftKeys, 32768, true, true);
//	ExtractSift(mSiftKeys, img, 5, 1.0f, 3.5f, 0.0f, false);
}

void Frame::SetPose(const Frame& frame) {
	mPose = frame.mPose;
	mPoseInv = frame.mPoseInv;
}

void Frame::release() {
	for(int i = 0; i < numPyrs; ++i) {
		mdIx[i].release();
		mdIy[i].release();
		mGray[i].release();
		mVMap[i].release();
		mNMap[i].release();
		mDepth[i].release();
		mDescriptors.release();
	}
}

Matrix3f Frame::Rot_gpu() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(mPose(0, 0), mPose(0, 1), mPose(0, 2));
	Rot.rowy = make_float3(mPose(1, 0), mPose(1, 1), mPose(1, 2));
	Rot.rowz = make_float3(mPose(2, 0), mPose(2, 1), mPose(2, 2));
	return Rot;
}

Matrix3f Frame::RotInv_gpu() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(mPoseInv(0, 0), mPoseInv(0, 1), mPoseInv(0, 2));
	Rot.rowy = make_float3(mPoseInv(1, 0), mPoseInv(1, 1), mPoseInv(1, 2));
	Rot.rowz = make_float3(mPoseInv(2, 0), mPoseInv(2, 1), mPoseInv(2, 2));
	return Rot;
}

float3 Frame::Trans_gpu() const {
	return make_float3(mPose(0, 3), mPose(1, 3), mPose(2, 3));
}

void Frame::SetPose(const Eigen::Matrix4d T) {
	mPose = T;
	Eigen::Matrix3d R = Rotation().transpose();
	Eigen::Vector3d t = -R * Translation();
	mPoseInv.topLeftCorner(3, 3) = R;
	mPoseInv.topRightCorner(3, 1) = t;
}

Eigen::Matrix3d Frame::Rotation() {
	return mPose.topLeftCorner(3, 3);
}

Eigen::Vector3d Frame::Translation() {
	return mPose.topRightCorner(3, 1);
}

void Frame::SetK(cv::Mat& K) {
	for(int i = 0; i < numPyrs; ++i) {
		mK[i] = cv::Mat::eye(3, 3, CV_32FC1);
		mK[i].at<float>(0, 0) = K.at<float>(0, 0) / (1 << i);
		mK[i].at<float>(1, 1) = K.at<float>(1, 1) / (1 << i);
		mK[i].at<float>(0, 2) = K.at<float>(0, 2) / (1 << i);
		mK[i].at<float>(1, 2) = K.at<float>(1, 2) / (1 << i);
	}
}

float Frame::fx(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mK[pyr].at<float>(0, 0);
}

float Frame::fy(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mK[pyr].at<float>(1, 1);
}

float Frame::cx(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mK[pyr].at<float>(0, 2);
}

float Frame::cy(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mK[pyr].at<float>(1, 2);
}

int Frame::cols(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mCols[pyr];
}

int Frame::rows(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mRows[pyr];
}

int Frame::pixels(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return N[pyr];
}

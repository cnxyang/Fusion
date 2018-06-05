#include "Frame.h"
#include "Converter.h"
#include "DeviceFunc.h"

int Frame::N[numPyrs];
cv::Mat Frame::mK[numPyrs];
bool Frame::mbFirstCall = true;
float Frame::mDepthCutoff = 5.0f;
float Frame::mDepthScale = 1000.0f;
std::pair<int, int> Frame::mPyrRes[numPyrs];
cv::Ptr<cv::cuda::ORB> Frame::mORB;

Frame::Frame() {}

Frame::~Frame() { release(); }

Frame::Frame(const Frame& other) {

	for(int i = 0; i < numPyrs; ++i) {
		other.mDepth[i].copyTo(mDepth[i]);
		other.mGray[i].copyTo(mGray[i]);
		other.mVMap[i].copyTo(mVMap[i]);
		other.mNMap[i].copyTo(mNMap[i]);
		other.mdIx[i].copyTo(mdIx[i]);
		other.mdIy[i].copyTo(mdIy[i]);
		other.mMapPoints.copyTo(mMapPoints);
	}

	mRcw = other.mRcw.clone();
	mtcw = other.mtcw.clone();
	mRwc = other.mRwc.clone();
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
//		other.mDepth[i].copyTo(mDepth[i]);
		other.mGray[i].copyTo(mGray[i]);
		other.mdIx[i].copyTo(mdIx[i]);
		other.mdIy[i].copyTo(mdIy[i]);
		other.mMapPoints.copyTo(mMapPoints);
	}

	mRcw = other.mRcw.clone();
	mtcw = other.mtcw.clone();
	mRwc = other.mRwc.clone();

	int p = 0;
	cv::Mat test(mdIx[p].rows(), mdIx[p].cols(), CV_32FC1);
	mdIx[p].download((void*)test.data, test.step);
	cv::imshow("test", test);
}


Frame::Frame(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& K) {

	if(mbFirstCall) {
		mORB = cv::cuda::ORB::create(1000);
		for(int i = 0; i < numPyrs; ++i) {

			mPyrRes[i].first = imD.cols / (1 << i);
			mPyrRes[i].second = imD.rows / (1 << i);
			N[i] = mPyrRes[i].first * mPyrRes[i].second;
//			mK[i] = cv::Mat::eye(3, 3, CV_32FC1);
//			mK[i].at<float>(0, 0) = K.at<float>(0, 0) / (1 << i);
//			mK[i].at<float>(1, 1) = K.at<float>(1, 1) / (1 << i);
//			mK[i].at<float>(0, 2) = K.at<float>(0, 2) / (1 << i);
//			mK[i].at<float>(1, 2) = K.at<float>(1, 2) / (1 << i);
		}
		mbFirstCall = false;
	}

	DeviceArray2D<uchar3> rawRGB(mPyrRes[0].first, mPyrRes[0].second);
	DeviceArray2D<ushort> rawDepth(mPyrRes[0].first, mPyrRes[0].second);
	rawRGB.upload((void*)imRGB.data, imRGB.step, mPyrRes[0].first, mPyrRes[0].second);
	rawDepth.upload((void*)imD.data, imD.step, mPyrRes[0].first, mPyrRes[0].second);
	for(int i = 0; i < numPyrs; ++i) {
		mdIx[i].create(mPyrRes[i].first, mPyrRes[i].second);
		mdIy[i].create(mPyrRes[i].first, mPyrRes[i].second);
		mGray[i].create(mPyrRes[i].first, mPyrRes[i].second);
		mVMap[i].create(mPyrRes[i].first, mPyrRes[i].second);
		mNMap[i].create(mPyrRes[i].first, mPyrRes[i].second);
		mDepth[i].create(mPyrRes[i].first, mPyrRes[i].second);
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

	cv::cuda::GpuMat GrayTemp, Descriptor;
	std::vector<cv::KeyPoint> mKeyPoints;
	GrayTemp.create(mPyrRes[0].second, mPyrRes[0].first, CV_8UC1);
	SafeCall(cudaMemcpy2D((void*)GrayTemp.data, GrayTemp.step,
				          (void*)mGray[0], mGray[0].step(), sizeof(char) * mGray[0].cols(),
				          mGray[0].rows(), cudaMemcpyDeviceToDevice));
	mORB->detectAndCompute(GrayTemp, cv::cuda::GpuMat(), mKeyPoints, Descriptor);
	mDescriptors.create(Descriptor.cols, Descriptor.rows);
	SafeCall(cudaMemcpy2D((void*)mDescriptors, mDescriptors.step(),
						  (void*)Descriptor.data, Descriptor.step, sizeof(char) * Descriptor.cols,
						  Descriptor.rows, cudaMemcpyDeviceToDevice));

//	mNkp = mKeyPoints.size();
//	if(mNkp > 0) {
//		float invfx = fx(0);
//		float invfy = fy(0);
//		float cx0 = cx(0);
//		float cy0 = cy(0);
//		mMapPoints.create(mNkp);
//		Point* pts = new Point[mNkp];
//		for(int i = 0; i < mNkp; ++i) {
//			cv::KeyPoint& kp = mKeyPoints[i];
//			Point& pt = pts[i];
//			float dp = (float)imD.at<uchar>(kp.pt.x, kp.pt.y) / mDepthScale;
//			if(dp > 1e-3 && dp < mDepthCutoff) {
//				pt.pos.z = dp;
//				pt.pos.x = dp * (kp.pt.x - cx0) * invfx;
//				pt.pos.y = dp * (kp.pt.y - cy0) * invfy;
//				pt.valid = true;
//			}
//			else
//				pt.valid = false;
//		}
//		mMapPoints.upload((void*)pts, mNkp);
//	}

	mRcw = cv::Mat::eye(3, 3, CV_32FC1);
	mtcw = cv::Mat::zeros(3, 1, CV_32FC1);
	mRwc = mRcw.t();
}

void Frame::SetPose(const Frame& frame) {
	mRcw = frame.mRcw.clone();
	mRwc = frame.mRwc.clone();
	mtcw = frame.mtcw.clone();
}

void Frame::release() {
	for(int i = 0; i < numPyrs; ++i) {
		mdIx[i].release();
		mdIy[i].release();
		mGray[i].release();
		mVMap[i].release();
		mNMap[i].release();
		mDepth[i].release();
		mMapPoints.release();
		mDescriptors.release();
	}
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
	return mPyrRes[pyr].first;
}

int Frame::rows(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return mPyrRes[pyr].second;
}

int Frame::pixels(int pyr) {
	assert(pyr >= 0 && pyr <= numPyrs);
	return N[pyr];
}

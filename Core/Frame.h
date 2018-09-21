#ifndef FRAME_HPP__
#define FRAME_HPP__

#include "DeviceMap.h"
#include "DeviceArray.h"
#include "KeyFrame.h"

#include <vector>
#include <opencv.hpp>
#include <features2d.hpp>
#include <cudaarithm.hpp>
#include <Eigen/Dense>

struct ORBKey;
struct KeyFrame;

struct Frame {

	Frame();
	Frame(Frame& other);
	Frame(const cv::Mat& imRGB, const cv::Mat& imD);
	Frame(const DeviceArray2D<uchar> & img, const cv::Mat & imD, KeyFrame * refKF);

	void SetPose(const Frame& frame);
	void SetPose(const Eigen::Matrix4d T);

	Eigen::Matrix3d Rotation() const;
	Eigen::Vector3d Translation();

	float3 Trans_gpu() const;
	Matrix3f Rot_gpu() const;
	Matrix3f RotInv_gpu() const;

	static const int NUM_PYRS = 3;
	static cv::Mat mK[NUM_PYRS];
	static int mCols[NUM_PYRS];
	static int mRows[NUM_PYRS];

	static float fx(int pyr);
	static float fy(int pyr);
	static float cx(int pyr);
	static float cy(int pyr);
	static int cols(int pyr);
	static int rows(int pyr);
	static int pixels(int pyr);
	static void SetK(cv::Mat& K);
	static bool mbFirstCall;
	static float mDepthScale;
	static float mDepthCutoff;
	static cv::Ptr<cv::cuda::ORB> mORB;

	std::vector<cv::Vec3f> mNormals;
	std::vector<Eigen::Vector3d> mPoints;

	cv::Mat rawDepth;

	int N;
	std::vector<cv::KeyPoint> keys;
	cv::cuda::GpuMat descriptors;

	Eigen::Matrix4d pose;

public:

	void setPose(Frame * other);
	void setPose(Eigen::Matrix4d & pose);
	Matrix3f absRotationCuda() const;
	Matrix3f absRotationInvCuda() const;
	float3 absTranslationCuda() const;
	Eigen::Matrix3d absRotation() const;
	Eigen::Vector3d absTranslation() const;

	KeyFrame * referenceKF;
	Eigen::Matrix4d deltaPose;

	unsigned long frameId;
	static unsigned long nextId;

	cv::Mat rawColor;
	cv::Mat scaledDepth;

	std::vector<int> index;
	std::vector<ORBKey> keyPoints;
};

#endif

#ifndef FRAME_HPP__
#define FRAME_HPP__

#include "devmap.h"
#include "cuarray.h"
#include "keyFrame.h"

#include <vector>
#include <opencv.hpp>
#include <features2d.hpp>
#include <cudaarithm.hpp>
#include <Eigen/Dense>

struct ORBKey;
struct KeyFrame;

class Frame {
public:
	Frame();
	Frame(const Frame& other);
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

public:

	static bool mbFirstCall;
	static float mDepthScale;
	static float mDepthCutoff;
	static cv::Ptr<cv::cuda::ORB> mORB;

	std::vector<cv::Vec3f> mNormals;
	std::vector<Eigen::Vector3d> mPoints;

	cv::Mat rawDepth;
	cv::Mat rawColor;
	cv::Mat scaledDepth;
	KeyFrame * referenceKF;
	unsigned long frameId;
	static unsigned long nextId;

	int N;
	std::vector<bool> outliers;
	std::vector<cv::KeyPoint> keys;
	cv::cuda::GpuMat descriptors;

	Eigen::Matrix4d pose;
};

#endif

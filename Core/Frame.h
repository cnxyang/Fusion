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
#include <xfeatures2d/cuda.hpp>

struct ORBKey;
struct KeyFrame;

struct Frame {

	Frame();
	Frame(Frame & other);
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



public:

	void setPose(Frame * other);
	void setPose(Eigen::Matrix4d & pose);
	Matrix3f absRotationCuda() const;
	Matrix3f absRotationInvCuda() const;
	float3 absTranslationCuda() const;
	Eigen::Matrix3d absRotation() const;
	Eigen::Vector3d absTranslation() const;

	Eigen::Matrix4d deltaPose;


	std::vector<int> index;
	std::vector<ORBKey> keyPoints;

public:

	void Create(int cols_, int rows_);
	void FillImages(const cv::Mat & range_, const cv::Mat & color_);
	void ExtractKeyPoints();
	void ResizeVNMap();

	DeviceArray2D<unsigned short> range;
	DeviceArray2D<uchar3> color;

	DeviceArray2D<float4> vmap[NUM_PYRS];
	DeviceArray2D<float4> nmap[NUM_PYRS];
	DeviceArray2D<float> depth[NUM_PYRS];
	DeviceArray2D<unsigned char> image[NUM_PYRS];

	cv::Mat rawColor;
	cv::Mat scaledDepth;
	unsigned long frameId;
	static unsigned long nextId;

	KeyFrame * referenceKF;
	Eigen::Matrix4f deltaT;
	Eigen::Matrix4d pose;

	int N;
	std::vector<float3> pt3d;
	std::vector<float4> normal;
	cv::cuda::GpuMat descriptors;
	std::vector<cv::KeyPoint> keys;

	static cv::cuda::SURF_CUDA surfExt;
	static cv::Ptr<cv::BRISK> briskExt;
};

#endif

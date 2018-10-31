#ifndef FRAME__
#define FRAME__

#include "KeyFrame.h"
#include "VectorMath.h"
#include "PoseStruct.h"
#include "KeyFrameGraph.h"
#include "DeviceArray.h"
#include <vector>
#include <opencv.hpp>
#include <sophus/se3.hpp>
#include <features2d.hpp>
#include <cudaarithm.hpp>
#include <xfeatures2d/cuda.hpp>

class Frame
{
public:

	static const int NUM_PYRS = 3;
	static const int MIN_KEY_POINTS = 500;

	Frame();
	Frame(const Frame * other);

	void Create(int cols_, int rows_);

	void ExtractKeyPoints();

	void ResizeImages();

	void Clear();

	void DrawKeyPoints();

	void ClearKeyPoints();

	void FillImages(const cv::Mat & range_, const cv::Mat & color_);

	float InterpDepth(cv::Mat & map, float & x, float & y);

	float4 InterpNormal(cv::Mat & map, float & x, float & y);

	Eigen::Matrix3d Rotation();

	Eigen::Matrix3d RotationInv();

	Eigen::Vector3d Translation();

	Eigen::Vector3d TranslationInv();

	Matrix3f GpuRotation();

	float3 GpuTranslation();

	Matrix3f GpuInvRotation();

	Eigen::Vector3f GetWorldPoint(int i);

	void operator=(const Frame & other);

	// =============== used for icp ====================
	// TODO: move this outside of the frame struct
	DeviceArray2D<unsigned short> temp;
	DeviceArray2D<float> range;
	DeviceArray2D<uchar3> color;

	DeviceArray2D<float4> vmap[NUM_PYRS];
	DeviceArray2D<float4> nmap[NUM_PYRS];
	DeviceArray2D<float> depth[NUM_PYRS];
	DeviceArray2D<unsigned char> image[NUM_PYRS];
	DeviceArray2D<short> dIdx[NUM_PYRS];
	DeviceArray2D<short> dIdy[NUM_PYRS];
	// =============== used for icp ====================


	// =============== key points ====================
	// TODO: get rid of this ASAP
	int N;
	bool bad;
	cv::cuda::GpuMat descriptors;
	std::vector<float4> pointNormal;
	std::vector<Eigen::Vector3f> mapPoints;
	std::vector<cv::KeyPoint> keyPoints;
	std::vector<bool> outliers;

	static cv::cuda::SURF_CUDA surfExt;
	static cv::Ptr<cv::BRISK> briskExt;
	// =============== key points ====================

	// =============== general information ====================

	unsigned long frameId;
	static unsigned long nextId;

	static Eigen::Matrix3f mK[NUM_PYRS];
	static int mCols[NUM_PYRS];
	static int mRows[NUM_PYRS];

	static float fx(int pyr);
	static float fy(int pyr);
	static float cx(int pyr);
	static float cy(int pyr);
	static int cols(int pyr);
	static int rows(int pyr);
	static void SetK(Eigen::Matrix3f& K);
	static bool mbFirstCall;
	static float mDepthScale;
	static float mDepthCutoff;

	// ===================== OLD JUNK ends here =====================

	// ==================== REFACTOR begins here ====================

	Frame(cv::Mat & image, cv::Mat & depth, int id, Eigen::Matrix3f K, double timeStamp);

	void initialize(int w, int h, int id, Eigen::Matrix3f & K, double timeStamp);

	/** Returns the unique frame id. */
	inline int id() const;
	/** Returns the frame's image width. */
	inline int width(int level = 0) const;
	/** Returns the frame's image height. */
	inline int height(int level = 0) const;
	/** Returns the frame's total number of pixels */
	inline int totalPixel(int level = 0) const;
	/** Returns the frame's intrinsics matrix. */
	inline const Eigen::Matrix3f & K(int level = 0) const;
	/** Returns the frame's inverse intrinsics matrix. */
	inline const Eigen::Matrix3f & KInv(int level = 0) const;
	/** Returns K(0, 0). */
	inline float getfx(int level = 0) const;
	/** Returns K(1, 1). */
	inline float getfy(int level = 0) const;
	/** Returns K(0, 2). */
	inline float getcx(int level = 0) const;
	/** Returns K(1, 2). */
	inline float getcy(int level = 0) const;
	/** Returns KInv(0, 0). */
	inline float fxInv(int level = 0) const;
	/** Returns KInv(1, 1). */
	inline float fyInv(int level = 0) const;
	/** Returns KInv(0, 2). */
	inline float cxInv(int level = 0) const;
	/** Returns KInv(1, 2). */
	inline float cyInv(int level = 0) const;
	/** Returns the frame's recording timestamp. */
	inline double timeStamp() const;

	inline Sophus::SE3d getCamToWorld() const;
	Sophus::SE3d pose;
	PoseStruct * poseStruct;

	struct Data {
		int id;
		int width[NUM_PYRS], height[NUM_PYRS];
		float fx[NUM_PYRS], fy[NUM_PYRS];
		float cx[NUM_PYRS], cy[NUM_PYRS];
		float fxInv[NUM_PYRS], fyInv[NUM_PYRS];
		float cxInv[NUM_PYRS], cyInv[NUM_PYRS];
		double timeStamp;

		Eigen::Matrix3f K[NUM_PYRS];
		Eigen::Matrix3f KInv[NUM_PYRS];

		cv::Mat image;
		cv::Mat depth;

	};

	Data data;

	std::unordered_map<Frame*, std::hash<Frame*>> neighbors;
};

inline Sophus::SE3d Frame::getCamToWorld() const
{
	return poseStruct->getCamToWorld();
}

inline int Frame::id() const
{
	return data.id;
}

inline int Frame::width(int level) const
{
	return data.width[level];
}

inline int Frame::height(int level) const
{
	return data.height[level];
}

inline int Frame::totalPixel(int level) const
{
	return data.width[level] * data.height[level];
}

inline const Eigen::Matrix3f& Frame::K(int level) const
{
	return data.K[level];
}

inline const Eigen::Matrix3f& Frame::KInv(int level) const
{
	return data.KInv[level];
}

inline float Frame::getfx(int level) const
{
	return data.fx[level];
}

inline float Frame::getfy(int level) const
{
	return data.fy[level];
}

inline float Frame::getcx(int level) const
{
	return data.cx[level];
}

inline float Frame::getcy(int level) const
{
	return data.cy[level];
}

inline float Frame::fxInv(int level) const
{
	return data.fxInv[level];
}

inline float Frame::fyInv(int level) const
{
	return data.fyInv[level];
}

inline float Frame::cxInv(int level) const
{
	return data.cxInv[level];
}

inline float Frame::cyInv(int level) const
{
	return data.cyInv[level];
}

inline double Frame::timeStamp() const
{
	return data.timeStamp;
}

#endif

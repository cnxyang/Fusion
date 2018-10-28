#ifndef TRACKING_H__
#define TRACKING_H__

#include <mutex>
#include <condition_variable>
#include "Frame.h"
#include "MapViewer.h"
#include "DenseMap.h"
#include "DeviceFuncs.h"

class MapViewer;
class DenseMap;

class Tracker {

public:

	Tracker();

	Tracker(int cols_, int rows_, float fx, float fy, float cx, float cy);

	bool GrabFrame(const cv::Mat & rgb, const cv::Mat & depth);

	void ResetTracking();

	void SetMap(DenseMap * map_);

	void SetViewer(MapViewer * viewer_);

	Eigen::Matrix4f GetCurrentPose() const;

	Intrinsics K;

	Sophus::SE3d nextPose;
	Sophus::SE3d lastPose;

	int state;
	int lastState;

	Frame * NextFrame;
	Frame * LastFrame;

	std::mutex updateImageMutex;
	std::atomic<bool> needImages;
	std::atomic<bool> imageUpdated;
	std::atomic<bool> mappingDisabled;
	std::atomic<bool> useGraphMatching;

	DeviceArray2D<uchar4> renderedImage;
	DeviceArray2D<uchar4> renderedDepth;
	DeviceArray2D<uchar4> rgbaImage;

	std::vector<Eigen::Vector3d> output;
	KeyFrame * ReferenceKF;
	KeyFrame * LastKeyFrame;

protected:

	bool Track();

	void SwapFrame();

	bool TrackFrame();

	void ComputeSO3();

	bool ComputeSE3(bool icpOnly, const int * iter, const float thresh_icp);

	void RenderView();

	bool TrackLastFrame();

	void CheckOutliers();

	bool Relocalise();

	bool GenerateGraph(int N);

	bool ValidatePose();

	void InitTracking();

	bool NeedKeyFrame();

	void CreateKeyFrame();

	void InsertFrame();

	DenseMap * map;

	MapViewer * viewer;

	const int maxIter = 35;
	const int maxIterReloc = 100;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

	int noInliers;
	int noMissedFrames;

	// ICP Tracking
	static const int NUM_PYRS = 3;
	const int ITERATIONS_SE3[NUM_PYRS] = { 10, 5, 3 };
	const int ITERATIONS_RELOC[NUM_PYRS] = { 3, 3, 2 };
	const int MIN_ICP_COUNT[NUM_PYRS] = { 2000, 1000, 100 };
	const float THRESH_ICP_SE3 = 0.0001f;
	const float THRESH_ICP_RELOC = 0.001f;

	float lastIcpError;
	float lastRgbError;

	// SO3 and SE3 residuals
	float icpResidual[2];
	float rgbResidual[2];
	float so3Residual[2];

	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSE3;
	DeviceArray<int> outRes;
	DeviceArray2D<int> sumRes;
	DeviceArray<float> outSO3;
	DeviceArray2D<float> sumSO3;

	bool mappingTurnedOff;
	std::vector<bool> outliers;
	std::vector<cv::DMatch> refined;

	// Graph based relocalization
	std::vector<Eigen::Vector3d> mapKeysAll;
	cv::cuda::GpuMat descriptors;

	const int N_LISTS_SELECT = 5;
	const int N_LISTS_SUB_GRAPH = 10;
	const int THRESH_N_SELECTION = 400;
	const float THRESH_MIN_SCORE = 0.1f;

	std::vector<SURF> frameKeys;
	std::vector<SURF> mapKeysMatched;
	std::vector<Eigen::Matrix4d> poseRefined;
	std::vector<Eigen::Matrix4d> poseEstimated;
	std::vector<std::vector<Eigen::Vector3d>> mapKeySelected;
	std::vector<std::vector<Eigen::Vector3d>> frameKeySelected;

	std::vector<Eigen::Vector3d> refPoints;
	std::vector<Eigen::Vector3d> framePoints;
	std::vector<float> distance;
	std::vector<int> queryKeyIdx;

	// Optimization
	std::vector<int> kfMatches;
};

#endif

#ifndef TRACKING_H__
#define TRACKING_H__

#include "Frame.h"
#include "Viewer.h"
#include "Mapping.h"
#include "Reduction.h"
#include <mutex>

class Viewer;
class Mapping;

class Tracker {

public:

	Tracker();

	Tracker(int cols_, int rows_, float fx, float fy, float cx, float cy);

	bool GrabFrame(const cv::Mat & rgb, const cv::Mat & depth);

	void ResetTracking();

	void SetMap(Mapping * map_);

	void SetViewer(Viewer * viewer_);

	Eigen::Matrix4f GetCurrentPose() const;

	Intrinsics K;

	Eigen::Matrix4d nextPose;
	Eigen::Matrix4d lastPose;

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

protected:

	bool Track();

	void SwapFrame();

	bool TrackFrame();

	void ComputeSO3();

	bool ComputeSE3();

	void RenderView();

	bool TrackReferenceKF();

	bool TrackReferenceKF_g2o();

	bool TrackLastFrame();

	bool TrackLastFrame_g2o();

	void CheckOutliers();

	bool Relocalise();

	void FilterMatching();

	void InitTracking();

	void FindNearestKF();

	bool NeedKeyFrame();

	void CreateKeyFrame();

	bool ValidateHypotheses();

	Mapping * map;
	Viewer * viewer;

	KeyFrame * ReferenceKF;
	KeyFrame * LastKeyFrame;

	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSE3;

	DeviceArray<int> outRes;
	DeviceArray2D<int> sumRes;

	DeviceArray<float> outSO3;
	DeviceArray2D<float> sumSO3;

	const int maxIter = 35;
	const int maxIterReloc = 100;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

	std::vector<Eigen::Vector3d> mapKeys;
	cv::cuda::GpuMat descriptors;

	int noInliers;
	int noMissedFrames;

	int iteration[3];
	int minIcpCount[3];

	float icpLastError;
	float rgbLastError;

	float icpResidual[2];
	float rgbResidual[2];
	float so3Residual[2];

	bool mappingTurnedOff;
	std::vector<bool> outliers;
	std::vector<cv::DMatch> refined;

	std::vector<SURF> frameKeySelected;
	std::vector<SURF> mapKeySelected;
	std::vector<float> matchDistance;
	std::vector<Eigen::Vector3d> refPoints;
	std::vector<Eigen::Vector3d> framePoints;
	std::vector<float> keyDistance;
	std::vector<int> queryKeyIdx;
};

#endif

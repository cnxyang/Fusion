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

protected:

	bool Track();

	void SwapFrame();

	bool TrackFrame();

	bool ComputeSE3();

	bool TrackReferenceKF();

	bool TrackLastFrame();

	bool Relocalise();

	void InitTracking();

	bool NeedKeyFrame();

	void CreateKeyFrame();

	Mapping * map;
	Viewer * viewer;

	KeyFrame * ReferenceKF;
	KeyFrame * LastKeyFrame;

	DeviceArray2D<float> sumSE3;
	DeviceArray2D<float> sumSO3;
	DeviceArray<float> outSE3;
	DeviceArray<float> outSO3;

	const int maxIter = 35;
	const int maxIterReloc = 100;
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

	std::vector<Eigen::Vector3d> mapKeys;
	cv::cuda::GpuMat descriptors;

	int noInliers;
	int noMissedFrames;

	int iteration[3];
	float icpResidual[2];
	float lastIcpError;
	float so3Residual[2];
	float lastSo3Error;

	std::vector<bool> outliers;
};

#endif

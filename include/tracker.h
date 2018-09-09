#ifndef TRACKING_HPP__
#define TRACKING_HPP__

#include "map.h"
#include "frame.h"
#include "viewer.h"
#include "cuarray.h"
#include "reduction.h"
#include <mutex>

class Viewer;
class Mapping;

class Tracker {
public:
	Tracker();
	Tracker(int w, int h, float fx, float fy, float cx, float cy);

	void reset();
	void initIcp();
	void swapFrame();
	void initTracking();
	void createNewKF();

	bool track();
	bool trackFrame(bool useKF = false);
	bool needNewKF();
	bool computeSO3();
	bool computeSE3();
	bool trackKeys();
	void fuseMapPoint();
	bool grabFrame(const cv::Mat & rgb, const cv::Mat & depth);
	bool relocalise();

	float rotationChanged() const;
	float translationChanged() const;

	void setMap(Mapping* pMap);
	void setTracker(Tracker * tracker);
	void setViewer(Viewer* pViewer);

	Eigen::Matrix4f getCurrentPose() const;

	MatK K;
	bool useIcp;
	bool useSo3;
	bool paused;
	static const int NUM_PYRS = 3;
	DeviceArray2D<unsigned short> depth;
	DeviceArray2D<uchar3> color;
	DeviceArray2D<uchar4> renderedImage;
	DeviceArray2D<uchar4> renderedDepth;
	DeviceArray2D<uchar4> rgbaImage;

	DeviceArray2D<float> lastDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> lastImage[NUM_PYRS];
	DeviceArray2D<float4> lastVMap[NUM_PYRS];
	DeviceArray2D<float3> lastNMap[NUM_PYRS];

	DeviceArray2D<float> nextDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
	DeviceArray2D<float4> nextVMap[NUM_PYRS];
	DeviceArray2D<float3> nextNMap[NUM_PYRS];
	DeviceArray2D<short> nextIdx[NUM_PYRS];
	DeviceArray2D<short> nextIdy[NUM_PYRS];

	DeviceArray<JtJJtrSE3> sumSE3;
	DeviceArray<JtJJtrSO3> sumSO3;
	DeviceArray<JtJJtrSE3> outSE3;
	DeviceArray<JtJJtrSO3> outSO3;

	KeyFrame * referenceKF;
	KeyFrame * lastKF;
	unsigned long lastReloc;

	Eigen::Matrix4d nextPose;
	Eigen::Matrix4d lastPose;
	Eigen::Matrix4d currentPose;
	Eigen::Matrix4d lastUpdatePose;

	int iteration[NUM_PYRS];
	float icpResidual[2];
	float lastIcpError;
	float rgbResidual[2];
	float lastRgbError;
	float so3Residual[2];
	float lastSo3Error;

	int state;
	int lastState;

	bool graphMatching;
	const int maxIter = 50;
	const int maxIterReloc = 200;
	int noAttempsBeforeReloc;
	std::vector<bool> outliers;
	cv::cuda::GpuMat keyDescriptors;
	std::vector<Eigen::Vector3d> mapPoints;
	cv::Ptr<cv::cuda::DescriptorMatcher> orbMatcher;

	Frame lastFrame;
	Frame nextFrame;
	Mapping* mpMap;
	Viewer* mpViewer;

	std::atomic<bool> imageUpdated;
	std::atomic<bool> needImages;
};

#endif

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
	void computeSO3();
	bool computeSE3();
	void fuseMapPoint();
	void extractFeatures();
	bool trackReferenceKF();
	bool trackLastFrame();
	void checkKeyPoints();
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
	DeviceArray2D<float4> lastNMap[NUM_PYRS];

	DeviceArray2D<float> nextDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
	DeviceArray2D<float4> nextVMap[NUM_PYRS];
	DeviceArray2D<float4> nextNMap[NUM_PYRS];
	DeviceArray2D<short> nextIdx[NUM_PYRS];
	DeviceArray2D<short> nextIdy[NUM_PYRS];

	DeviceArray2D<float> sumSE3;
	DeviceArray<float> outSE3;
	DeviceArray2D<float> sumSO3;
	DeviceArray<float> outSO3;

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
	float so3Residual[2];
	float lastSo3Error;

	int state;
	int lastState;

	int N;
	std::atomic<bool> localisationOnly;
	std::atomic<bool> graphMatching;
	const int maxIter = 50;
	const int maxIterReloc = 200;
	int noAttempsBeforeReloc;
	unsigned long int lastRelocId;
	std::vector<bool> outliers;
	cv::cuda::GpuMat keyDescriptors;
	std::vector<Eigen::Vector3d> mapPoints;
	cv::Ptr<cv::cuda::DescriptorMatcher> orbMatcher;
	cv::Ptr<cv::cuda::ORB> orbExtractor;

	Frame lastFrame;
	Frame nextFrame;
	Mapping * mpMap;
	Viewer * mpViewer;

	std::atomic<bool> imageUpdated;
	std::atomic<bool> needImages;
};

#endif

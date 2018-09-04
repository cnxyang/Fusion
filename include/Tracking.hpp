#ifndef TRACKING_HPP__
#define TRACKING_HPP__

#include "reduction.h"
#include "device_map.hpp"
#include "Mapping.hpp"
#include "Viewer.hpp"
#include "Frame.hpp"
#include <vector>

class Viewer;
class Mapping;

using namespace cv;

class Tracking {
public:
	Tracking();
	Tracking(int w, int h, float fx, float fy, float cx, float cy);
	void SetMap(Mapping* pMap);
	void SetViewer(Viewer* pViewer);
	bool Track(Mat& imRGB, Mat& imD);
	void ResetTracking();

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	bool TrackMap(bool bUseGraph = true);
	bool TrackFrame();
	bool InitTracking();
	void SetState(State s);

	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;
	State mLastState;
	Mapping* mpMap;
	Viewer* mpViewer;

	cv::Mat desc;
	uint mnMapPoints;
	bool mbGraphMatching;
	int mnNoAttempts;
	const float mRotThresh = 0.2;
	const float mTransThresh = 0.05;
	DeviceArray<ORBKey> mDeviceKeys;
	std::vector<ORBKey> mHostKeys;
	std::vector<Eigen::Vector3d> mMapPoints;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;

public:

	void reset();
	void initIcp();
	void swapFrame();
	void initTracking();
	void createNewKF();

	bool track();
	bool trackFrame();
	bool trackKF();
	bool needNewKF();
	bool computeSO3();
	bool computeSE3();
	bool trackKeys();
	bool grabFrame(cv::Mat & rgb, cv::Mat & depth);
	bool relocalise();

	MatK K;
	bool useIcp;
	bool useSo3;
	bool graphMatching;
	static const int NUM_PYRS = 3;
	DeviceArray2D<unsigned short> depth;
	DeviceArray2D<uchar3> color;

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

	Frame * nextFrame;
	Frame * lastFrame;

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
};

#endif

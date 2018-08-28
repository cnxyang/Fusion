#ifndef TRACKING_HPP__
#define TRACKING_HPP__

#include "device_map.hpp"
#include "Mapping.hpp"
#include "Viewer.hpp"
#include "Frame.hpp"
#include <vector>

class Viewer;

using namespace cv;

class Tracking {
public:
	Tracking();
	void SetMap(Mapping* pMap);
	void SetViewer(Viewer* pViewer);
	bool Track(Mat& imRGB, Mat& imD);
	void ResetTracking();
	void AddObservation(const Rendering& render);

public:

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	bool TrackMap(bool bUseGraph = true);
	bool TrackICP();
	bool TrackFrame();
	bool InitTracking();
	void UpdateMap();
	void UpdateFrame();
	void SetState(State s);
	bool TrackLastFrame();
	void ShowResiduals();

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
};

#endif

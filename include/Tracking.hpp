#ifndef TRACKING_HPP__
#define TRACKING_HPP__

#include "device_map.cuh"
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
	bool TrackMap();
	bool TrackICP();
	bool TrackFrame();
	bool InitTracking();
	void UpdateMap();
	void UpdateFrame();
	bool TrackLastFrame();
	void ShowResiduals();

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;
	Mapping* mpMap;
	Viewer* mpViewer;

	int mnMapPoints;
	const float mRotThresh = 0.1;
	const float mTransThresh = 0.05;
	DeviceArray<ORBKey> mMapPoints;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;
};

#endif

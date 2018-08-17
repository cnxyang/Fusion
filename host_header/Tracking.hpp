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
	void AddObservation(const Rendering& render);

public:
	bool TrackMap();
	void TrackICP();
	bool TrackFrame();
	bool InitTracking();
	void UpdateMap();
	void UpdateFrame();
	bool TrackLastFrame();

	/* For debugging purposes */
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

	static bool mbTrackModel;
	int mnMapPoints;
	DeviceArray<ORBKey> mMapPoints;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;
};

#endif

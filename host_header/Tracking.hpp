#ifndef TRACKING_HPP__
#define TRACKING_HPP__

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
	void TrackICP();
	bool TrackFrame();
	bool CreateInitialMap();
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
	Ptr<cuda::DescriptorMatcher> mORBMatcher;
};

#endif

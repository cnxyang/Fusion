#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Map.hpp"
#include "Frame.hpp"
#include <vector>

using namespace cv;

class Tracking {
public:
	Tracking();
	void SetMap(Map* pMap);
	bool GrabImageRGBD(Mat& imRGB, Mat& imD);
	void AddObservation(const Rendering& render);

public:
	bool Track();
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
	Mat mK;
	Map* mpMap;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;
};

#endif

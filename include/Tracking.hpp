#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Mapping.hpp"
#include "Frame.hpp"
#include <vector>

using namespace cv;

class Tracking {
public:
	Tracking();
	void SetMap(Mapping* pMap);
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
	Mat mK;
	Mapping* mpMap;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;
};

#endif

#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Map.h"
#include "Frame.h"
#include <vector>

class Tracking {
public:
	Tracking();
	void GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);
	void SetObservation(const Rendering& render);

public:
	void Track();
	void TrackMap();
	void TrackICP();
	bool InitTracking();
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
	cv::Mat mK;
	Map* mpMap;
};

#endif

#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Map.h"
#include "Frame.h"
#include <vector>

class Tracking {
public:
	Tracking();
	void GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);
	void AddObservation(const Rendering& render);

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

	float cost;
	const int iter[3] = { 10, 5, 3 };
	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;
	cv::Mat mK;
	Map* mpMap;
};

#endif

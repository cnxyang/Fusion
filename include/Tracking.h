#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Frame.h"

class Tracking {
public:
	Tracking();
	void GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);

private:
	void Track();
	bool InitTracking();
	bool TrackLastFrame();
	void VisualiseTrackingResult();

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;

	cv::Mat mK;
};

#endif

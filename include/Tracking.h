#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Frame.h"
#include <vector>

class Tracking {
public:
	Tracking();
	void GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);
	std::vector<cv::Mat>& GetPoses() { return Poses; }

private:
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

	std::vector<cv::Mat> Poses;

	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;

	cv::Mat mK;
};

#endif

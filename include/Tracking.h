#ifndef __TRACKING_H__
#define __TRACKING_H__

#include "Map.h"
#include "Frame.h"
#include <vector>

class Tracking {
public:
	Tracking();
	void SetMap(Map* pMap);
	bool GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD);
	void AddObservation(const Rendering& render);

public:
	bool Track();
	bool TrackMap();
	void TrackICP();
	bool TrackFrame();
	bool InitTracking();
	void NeedNewKeyFrame();
	void CreateKeyFrame();
	bool Relocalisation();
	bool TrackLastFrame();
	void ShowResiduals();

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	int mNoFrames;
	bool mbNeedNewKF;
	float cost;
	const int iter[3] = { 10, 5, 3 };
	Frame mLastFrame;
	Frame mNextFrame;
	KeyFrame mLastKeyFrame;
	State mNextState;
	cv::Mat mK;
	Map* mpMap;
	cv::Ptr<cv::cuda::DescriptorMatcher> mORBMatcher;
};

#endif

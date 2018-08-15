#ifndef __CAMERA__
#define __CAMERA__

#include <OpenNI.h>
#include <opencv2/opencv.hpp>

using openni::Device;
using openni::VideoStream;
using openni::VideoFrameRef;

class CameraNI
{
public:
	CameraNI();
	CameraNI(int, int, int);
	~CameraNI();

	void InitCamera();
	void StartStreaming();
	void StopStreaming();
	bool FetchFrame(cv::Mat&, cv::Mat&);
	int GetImgRows() const;
	int GetImgCols() const;
	int GetFPS() const;

private:
	void FetchColorFrame(cv::Mat&);
	void FetchDepthFrame(cv::Mat&);

	int mCols, mRows, mFPS;
	Device* 			mpDevice;
	VideoStream* 		mpColorStream;
	VideoStream* 		mpDepthStream;
	VideoFrameRef* 		mpColorFrame;
	VideoFrameRef* 		mpDepthFrame;
};

#endif

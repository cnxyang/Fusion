#ifndef CAMERA_HPP__
#define CAMERA_HPP__

#include <OpenNI.h>
#include <opencv.hpp>

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

	int rows() const;
	int cols() const;
	int fps() const;

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

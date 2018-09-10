#ifndef CAMERA_HPP__
#define CAMERA_HPP__

#include <OpenNI.h>
#include <opencv.hpp>

using openni::Device;
using openni::VideoStream;
using openni::VideoFrameRef;

class camera
{
public:
	camera();
	camera(int, int, int);
	~camera();

	void initCamera();
	void startStreaming();
	void stopStreaming();
	bool fetchFrame(cv::Mat&, cv::Mat&);

	int rows() const;
	int cols() const;
	int fps() const;

private:

	void fetchColorFrame(cv::Mat&);
	void fetchDepthFrame(cv::Mat&);
	int mCols, mRows, mFPS;

	Device* 			mpDevice;
	VideoStream* 		mpColorStream;
	VideoStream* 		mpDepthStream;
	VideoFrameRef* 		mpColorFrame;
	VideoFrameRef* 		mpDepthFrame;
};

#endif

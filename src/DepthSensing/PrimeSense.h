#pragma once
#include <OpenNI.h>
#include <opencv.hpp>

using openni::Device;
using openni::VideoStream;
using openni::VideoFrameRef;

class PrimeSense
{
public:

	PrimeSense();
	PrimeSense(int w, int h, int fps);
	~PrimeSense();

	void initialization();
	void startStreaming();
	void stopStreaming();
	bool fetchFrame(cv::Mat& depth, cv::Mat& rgb);

	void setAutoExposure(bool value);
	void setAutoWhiteBalance(bool value);

protected:

	void fetchRGBFrame(cv::Mat& rgb);
	void fetchDepthFrame(cv::Mat& depth);

	openni::Device* device;
	openni::VideoStream* colorStream;
	openni::VideoStream* depthStream;
	openni::VideoFrameRef* colorFrameRef;
	openni::VideoFrameRef* depthFrameRef;
	openni::CameraSettings* settings;

	int width;
	int height;
	int frameRate;
};

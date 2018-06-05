/*
 * CameraNI.h
 *
 *  Created on: 18 Feb 2018
 *      Author: xy
 */

#ifndef CameraNI_H
#define CameraNI_H

#include <OpenNI.h>
#include <opencv2/opencv.hpp>

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
	openni::Device * mpDevice;
	openni::VideoStream * mpColorStream;
	openni::VideoStream * mpDepthStream;
	openni::VideoFrameRef * mpColorFrame;
	openni::VideoFrameRef * mpDepthFrame;
};

#endif

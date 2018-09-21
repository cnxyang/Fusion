#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv.hpp>
#include <highgui.hpp>

#include "Camera.h"
#include "Tracking.h"

int main(int argc, char** argv) {

	SysDesc desc;
	camera cam;
	cv::Mat imD, imRGB;

	desc.DepthCutoff = 3.0f;
	desc.DepthScale = 1000.0f;
	desc.cols = 640;
	desc.rows = 480;
	desc.fx = 520.149963;
	desc.fy = 516.175781;
	desc.cx = 309.993548;
	desc.cy = 227.090932;
	desc.TrackModel = true;
	desc.bUseDataset = false;

	System slam(&desc);

	while (1) {
		if (cam.fetchFrame(imD, imRGB)) {
			bool valid = slam.grabImage(imRGB, imD);
			if(!valid) {
				cam.stopStreaming();
				return 0;
			}
		}
	}
}

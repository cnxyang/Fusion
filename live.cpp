#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "Camera.hpp"
#include "Tracking.hpp"
#include "Timer.hpp"

int main(int argc, char** argv) {

	SysDesc desc;
	CameraNI camera;
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
		if (camera.FetchFrame(imD, imRGB)) {
			slam.GrabImageRGBD(imRGB, imD);
			slam.PrintTimings();
			int key = cv::waitKey(10);
			switch (key) {

			case 27: /* Escape */
				camera.StopStreaming();
				std::cout << "User Requested Termination." << std::endl;
				exit(0);
			}
		}
	}
}

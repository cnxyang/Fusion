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

enum MemRepType {
	Byte = 1, KiloByte = 1024, MegaByte = 1024 * 1024, GigaByte = 1024 * 1024 * 1024
};

void CudaCheckMemory(float & free, float & total, const MemRepType factor) {
	size_t freeByte;
	size_t totalByte;
	SafeCall(cudaMemGetInfo(&freeByte, &totalByte));
	free = (float) freeByte / factor;
	total = (float) totalByte / factor;
}

void PrintMemoryConsumption();

int main(int argc, char ** argv) {

	CameraNI camera;
	cv::Mat imD, imRGB;
	Tracking Tracker;
	Mapping map;
	Tracker.SetMap(&map);
	map.AllocateDeviceMemory(MapDesc());
	map.ResetDeviceMemory();
	cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);

	K.at<float>(0, 0) = 520.149963;
	K.at<float>(1, 1) = 516.175781;
	K.at<float>(0, 2) = 309.993548;
	K.at<float>(1, 2) = 227.090932;
	Frame::SetK(K);
	Frame::mDepthScale = 1000.0f;

	while (1) {
		auto t1 = std::chrono::steady_clock::now();
		if (camera.FetchFrame(imD, imRGB)) {

			cv::imshow("depth", imD);
			cv::imshow("rgb", imRGB);
			bool bOK = Tracker.Track(imRGB, imD);
			if (bOK) {
				int no = map.FuseFrame(Tracker.mLastFrame);
				Rendering rd;
				rd.cols = 640;
				rd.rows = 480;
				rd.fx = K.at<float>(0, 0);
				rd.fy = K.at<float>(1, 1);
				rd.cx = K.at<float>(0, 2);
				rd.cy = K.at<float>(1, 2);
				rd.Rview = Tracker.mLastFrame.Rot_gpu();
				rd.invRview = Tracker.mLastFrame.RotInv_gpu();
				rd.maxD = 5.0f;
				rd.minD = 0.1f;
				rd.tview = Tracker.mLastFrame.Trans_gpu();

				map.RenderMap(rd, no);
				Tracker.AddObservation(rd);
				cv::Mat tmp(rd.rows, rd.cols, CV_8UC4);
				rd.Render.download((void*) tmp.data, tmp.step);
				cv::resize(tmp, tmp, cv::Size(tmp.cols * 2, tmp.rows * 2));
				cv::imshow("img", tmp);
				cv::imshow("depth", imD);
			}

//			Timer::PrintTiming();
		}
		auto t2 = std::chrono::steady_clock::now();

		int key = cv::waitKey(10);
		switch (key) {

		case 't':
		case 'T':
			std::cout << "----------------------------------------------\n"
					<< "Frame Processed in : "
					<< std::chrono::duration_cast<std::chrono::microseconds>(
							t2 - t1).count() << " microseconds" << std::endl;
			break;

		case 'm':
		case 'M':
			PrintMemoryConsumption();
			break;

		case 27: /* Escape */
			camera.StopStreaming();
			std::cout << "User Requested Termination." << std::endl;
			exit(0);
		}
	}

	std::cout << "Finished Processing Dataset, awaiting cleanup process."
			<< std::endl;
	std::cout << "Program finished, exiting." << std::endl;
}

void PrintMemoryConsumption() {
	float free, total;
	CudaCheckMemory(free, total, MemRepType::MegaByte);
	std::cout << "----------------------------------------------\n"
			<< "Device Memory Consumption:\n" << "Free  - " << free << "MB\n"
			<< "Total - " << total << "MB" << std::endl;
}

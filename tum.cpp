#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "frame.h"
#include "tracker.h"

enum MemRepType {
	Byte = 1,
	KiloByte = 1024,
	MegaByte = 1024 * 1024,
	GigaByte = 1024 * 1024 * 1024
};

void CudaCheckMemory(float & free, float & total, const MemRepType factor) {
	size_t freeByte;
	size_t totalByte;
	SafeCall(cudaMemGetInfo(&freeByte, &totalByte));
	free = (float) freeByte / factor;
	total = (float) totalByte / factor;
}

void PrintMemoryConsumption();
void LoadDatasetTUM(std::string & sRootPath, std::vector<std::string> & vsDList,
		std::vector<std::string> & vsRGBList,
		std::vector<double> & vdTimeStamp);

int main(int argc, char** argv) {

	if (argc != 2) {
		std::cout << "Wrong Parameters.\n"
				<< "Usage: ./tum_example path_to_tum_dataset" << std::endl;
		exit(-1);
	}

	std::string sPath = std::string(argv[1]);
	std::vector<std::string> vsDList;
	std::vector<std::string> vsRGBList;
	std::vector<double> vdTimeList;

	LoadDatasetTUM(sPath, vsDList, vsRGBList, vdTimeList);

	if (vsDList.empty()) {
		std::cout << "Error occurs while reading the dataset.\n"
				<< "Please check your input parameters." << std::endl;
		exit(-1);
	}

	SysDesc desc;

	desc.DepthCutoff = 8.0f;
	desc.DepthScale = 5000.0f;
	desc.cols = 640;
	desc.rows = 480;
	desc.fx = 517.3;
	desc.fy = 516.5;
	desc.cx = 318.6;
	desc.cy = 255.3;
	desc.TrackModel = true;
	desc.bUseDataset = false;

	System slam(&desc);

	int nImages = std::min(vsRGBList.size(), vdTimeList.size());
	std::cout << "----------------------------------------------\n"
			<< "Total Images to be processed: " << nImages << std::endl;
	for (int i = 0; i < nImages; ++i) {
		cv::Mat imD = cv::imread(vsDList[i], cv::IMREAD_UNCHANGED);
		cv::Mat imRGB = cv::imread(vsRGBList[i], cv::IMREAD_UNCHANGED);
		bool nonstop = slam.grabImage(imRGB, imD);
		if(!nonstop)
			return 0;
	}

	slam.joinViewer();
}

void LoadDatasetTUM(std::string & sRootPath, std::vector<std::string> & vsDList,
		std::vector<std::string> & vsRGBList,
		std::vector<double> & vdTimeList) {

	std::ifstream dfile, rfile;
	std::string sDPath, sRGBPath;

	if (sRootPath.back() != '/')
		sRootPath += '/';

	dfile.open(sRootPath + "depth.txt");
	rfile.open(sRootPath + "rgb.txt");

	std::string temp;
	for (int i = 0; i < 3; ++i) {
		getline(dfile, temp);
		getline(rfile, temp);
	}

	std::string line, sD, sR;
	while (true) {
		double t;
		dfile >> t;
		vdTimeList.push_back(t);
		dfile >> sD;
		vsDList.push_back(sRootPath + sD);
		rfile >> sR;
		rfile >> sR;
		vsRGBList.push_back(sRootPath + sR);
		if (dfile.eof() || rfile.eof())
			return;
	}
}

void PrintMemoryConsumption() {
	float free, total;
	CudaCheckMemory(free, total, MemRepType::MegaByte);
	std::cout << "----------------------------------------------\n"
			<< "Device Memory Consumption:\n" << "Free  - " << free << "MB\n"
			<< "Total - " << total << "MB" << std::endl;
}

#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "Converter.h"
#include "Tracking.h"

void SaveTrajectoryTUM(const std::string& filename, Tracking* pTracker, std::vector<double>& vdTimeList);
void LoadDatasetTUM(std::string & sRootPath,
					std::vector<std::string> & vsDList,
					std::vector<std::string> & vsRGBList,
					std::vector<double> & vdTimeStamp);

int main(int argc, char ** argv) {
//	std::cout << std::fixed;
//	std::cout << std::setprecision(4);

	if(argc != 2) {
		std::cout << "Wrong Parameters.\n"
				  << "Usage: ./tum_example path_to_tum_dataset" << std::endl;
		exit(-1);
	}

	std::string sPath = std::string(argv[1]);
	std::vector<std::string> vsDList;
	std::vector<std::string> vsRGBList;
	std::vector<double> vdTimeList;

	LoadDatasetTUM(sPath, vsDList, vsRGBList, vdTimeList);

	if(vsDList.empty()) {
		std::cout << "Error occurs while reading the dataset.\n"
				  << "Please check your input parameters." << std::endl;
		exit(-1);
	}

	Tracking Tracker;
	int nImages = std::min(vsRGBList.size(), vdTimeList.size());
	std::cout << "----------------------------------------------\n"
			  << "Total Images to be processed: " << nImages << std::endl;
	for(int i = 0; i < nImages; ++i) {
		cv::Mat imD = cv::imread(vsDList[i], cv::IMREAD_UNCHANGED);
		cv::Mat imRGB = cv::imread(vsRGBList[i], cv::IMREAD_UNCHANGED);

		Tracker.GrabImageRGBD(imRGB, imD);
        int key = cv::waitKey(10);

		switch(key) {
		case 27: /* Escape */
			std::cout << "User Requested Termination." << std::endl;
			exit(0);
		}
	}

	std::cout << "Finished Processing Dataset, awaiting cleanup process." << std::endl;
	std::cout << "Save Trajectories? (Y/N)." << std::endl;
	int key = cv::waitKey(15 * 1000);
	if(key == 'y' || key == 'Y') {
		SaveTrajectoryTUM("./1.txt", &Tracker, vdTimeList);
	}
	std::cout << "Program finished, exiting." << std::endl;
}

void LoadDatasetTUM(std::string & sRootPath,
					std::vector<std::string> & vsDList,
					std::vector<std::string> & vsRGBList,
					std::vector<double> & vdTimeList) {

	std::ifstream dfile, rfile;
	std::string sDPath, sRGBPath;

	if(sRootPath.back() != '/')
		sRootPath += '/';

	dfile.open(sRootPath + "depth.txt");
	rfile.open(sRootPath + "rgb.txt");

	std::string temp;
	for(int i = 0; i < 3; ++i) {
		getline(dfile, temp);
		getline(rfile, temp);
	}

	std::string line, sD, sR;
	while(true) {
		double t;
		dfile >> t;
		vdTimeList.push_back(t);
		dfile >> sD;
		vsDList.push_back(sRootPath + sD);
		rfile >> sR;
		rfile >> sR;
		vsRGBList.push_back(sRootPath + sR);
		if(dfile.eof() || rfile.eof()) return;
	}
}

void SaveTrajectoryTUM(const std::string& filename, Tracking* pTracker, std::vector<double>& vdTimeList) {

	std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
    std::vector<cv::Mat>& Poses = pTracker->GetPoses();
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    std::vector<double>::iterator lT = vdTimeList.begin();
    for(std::vector<cv::Mat>::iterator lit = Poses.begin(), lend = Poses.end(); lit != lend ;lit++, lT++) {
        cv::Mat Rwc = (*lit);
        cv::Mat twc = (*++lit);
        std::cout << Rwc << std::endl << twc << std::endl;

        std::vector<float> q = Converter::toQuaternion(Rwc);

        f << std::setprecision(6) << *lT << " "
        		                  <<  std::setprecision(9) << twc.at<float>(0) << " "
        		                  << twc.at<float>(1) << " "
        		                  << twc.at<float>(2) << " "
        		                  << q[0] << " "
        		                  << q[1] << " "
        		                  << q[2] << " "
        		                  << q[3] << std::endl;
    }

    f.close();
    std::cout << std::endl << "trajectory saved!" << std::endl;
}

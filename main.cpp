#include "System.hpp"
#include "opencv.hpp"
#include "RealSense.hpp"

using namespace cv;
using namespace std;

int main2(int argc, char** argv) {

	SysDesc desc;
	RealSense rs(640, 480);

	desc.DepthCutoff = 8.0f;
	desc.DepthScale = 1000.0f;
	desc.cols = 640;
	desc.rows = 480;
	desc.fx = rs.fx;
	desc.fy = rs.fy;
	desc.cx = rs.cx;
	desc.cy = rs.cy;
	desc.TrackModel = true;
	desc.bUseDataset = false;

	System slam(&desc);
	void* color = malloc(sizeof(uchar)*3*640*480);
	void* depth = malloc(sizeof(ushort)*640*480);

	while(1) {
		if(rs.Poll_for_frame(color, depth)) {
			Mat rgb(480, 640, CV_8UC3, color);
			Mat d(480, 640, CV_16UC1, depth);
			imshow("rgb", rgb);
			imshow("d", d);
			int key = waitKey(10);
			if(key == 27)
				return 0;

			slam.GrabImageRGBD(rgb, d);
		}
	}
}

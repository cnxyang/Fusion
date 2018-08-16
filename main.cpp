#include "System.hpp"
#include "opencv.hpp"
#include "Camera.hpp"

using namespace cv;
using namespace std;

typedef CameraNI Camera;

int main2(int argc, char** argv) {

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
	Camera cam;
	Mat imD, imRGB;

	while(1) {
		if(cam.FetchFrame(imD, imRGB)) {
			slam.GrabImageRGBD(imRGB, imD);
		}
	}
}

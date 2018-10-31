#include "Camera.h"
#include "SlamSystem.h"

int main(int argc, char** argv)
{
	PrimeSense cam;
	cv::Mat imD, imRGB;

	Eigen::Matrix3f K;

	K << 520.149963, 0.f, 309.993548,
		 0.f, 516.17578, 227.090932,
		 0.f, 0.f, 1.f;

	SlamSystem slam(640, 480, K);
//	cam.SetAutoExposure(false);
//	cam.SetAutoWhiteBalance(false);

	int i = 0;
	while (!slam.shouldQuit())
	{
		if (cam.FetchFrame(imD, imRGB))
		{
			slam.trackFrame(imRGB, imD, i++, 0);
		}
	}
}

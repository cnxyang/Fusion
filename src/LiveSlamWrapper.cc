#include "PrimeSense.h"
#include "SlamSystem.h"

int main(int argc, char** argv)
{
	PrimeSense depthCamera;
	cv::Mat imD, imRGB;

	Eigen::Matrix3f K;

	K << 520.149963, 0.f, 309.993548,
		 0.f, 516.17578, 227.090932,
		 0.f, 0.f, 1.f;

	SlamSystem slam(640, 480, K);

	int i = 0;
	while (!slam.shouldQuit())
	{
		if (depthCamera.fetchFrame(imD, imRGB))
		{
			slam.trackFrame(imRGB, imD, i++, 0);
		}
	}

	depthCamera.stopStreaming();
}

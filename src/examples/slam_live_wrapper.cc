#include "PrimeSense.h"
#include "SlamSystem.h"

int main(int argc, char** argv)
{
	PrimeSense depthCamera;
	cv::Mat imD, imRGB;

	Eigen::Matrix3f K;

	K << 528, 0.f, 320,
		 0.f, 528, 240,
		 0.f, 0.f, 1.f;

	SlamSystem slam(640, 480, K);

	int i = 0;
	cv::Mat depth_float;
	while (!slam.shouldQuit())
	{
		if (depthCamera.fetchFrame(imD, imRGB))
		{
			imD.convertTo(depth_float, CV_32FC1, 1.0f / 1000.f);
			slam.trackFrame(imRGB, depth_float, i++, 0);
		}
	}

	depthCamera.stopStreaming();
}

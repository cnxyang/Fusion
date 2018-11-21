#include "PrimeSense.h"
#include "SlamSystem.h"

int main(int argc, char** argv)
{
	PrimeSense depthCamera;
	cv::Mat imD, imRGB;

	Eigen::Matrix3f K;

	K << 571.7359619140625, 0.f, 327.1837811580408,
		 0.f, 573.6942749023438, 252.8661143421305,
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

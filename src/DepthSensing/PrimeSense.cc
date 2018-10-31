#include "PrimeSense.h"

PrimeSense::PrimeSense() :
	PrimeSense(640, 480, 30) {}

PrimeSense::PrimeSense(int cols, int rows, int fps) :
	width(cols), height(rows), frameRate(fps), device(0),
	colorStream(0), colorFrameRef(0), depthStream(0),
	depthFrameRef(0), settings(0)
{
	initialization();
	startStreaming();
}

PrimeSense::~PrimeSense()
{
	stopStreaming();
}

void PrimeSense::setAutoExposure(bool value)
{
	if(!settings)
		settings = colorStream->getCameraSettings();

	settings->setAutoExposureEnabled(value);
}

void PrimeSense::setAutoWhiteBalance(bool value)
{
	if(!settings)
		settings = colorStream->getCameraSettings();

	settings->setAutoWhiteBalanceEnabled(value);
}

void PrimeSense::initialization()
{

	if (openni::OpenNI::initialize() != openni::STATUS_OK)
	{
		printf("OpenNI Initialisation Failed with Error Message : %s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	device = new openni::Device();
	if (device->open(openni::ANY_DEVICE) != openni::STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	depthStream = new openni::VideoStream();
	colorStream = new openni::VideoStream();
	if (depthStream->create(*device, openni::SENSOR_DEPTH) != openni::STATUS_OK	||
	    colorStream->create(*device, openni::SENSOR_COLOR) != openni::STATUS_OK)
	{
		printf("Couldn't create streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	openni::VideoMode depth_video_mode = depthStream->getVideoMode();
	depth_video_mode.setResolution(width, height);
	depth_video_mode.setFps(frameRate);
	depth_video_mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

	openni::VideoMode color_video_mode = colorStream->getVideoMode();
	color_video_mode.setResolution(width, height);
	color_video_mode.setFps(frameRate);
	color_video_mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	// save customised mode
	depthStream->setVideoMode(depth_video_mode);
	colorStream->setVideoMode(color_video_mode);

	// Note: Doing image registration earlier than this point seems to fail
	if (device->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
	{
		if (device->setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) == openni::STATUS_OK)
		{
			printf("Depth To Colour Image Registration Set Success\n");
		}
		else
		{
			printf("Depth To Colour Image Registration Set FAILED\n");
		}
	}
	else
	{
		printf("Depth To Colour Image Registration is NOT Supported!!!\n");
	}

	printf("OpenNI Camera Initialisation Complete!\n");
}

void PrimeSense::startStreaming()
{

	depthStream->setMirroringEnabled(false);
	colorStream->setMirroringEnabled(false);

	if (depthStream->start() != openni::STATUS_OK)
	{
		printf("Couldn't start depth streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	if (colorStream->start() != openni::STATUS_OK)
	{
		printf("Couldn't start colour streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	depthFrameRef = new openni::VideoFrameRef();
	colorFrameRef = new openni::VideoFrameRef();

	printf("Camera Stream Started!\n");
}

void PrimeSense::stopStreaming()
{

	depthStream->stop();
	colorStream->stop();

	depthStream->destroy();
	colorStream->destroy();

	device->close();

	openni::OpenNI::shutdown();
	printf("Camera Stream Successfully Stopped.\n");
}

bool PrimeSense::fetchFrame(cv::Mat& depth, cv::Mat& rgb)
{

	openni::VideoStream * streams[] = { depthStream, colorStream };
	int streamReady = -1;
	auto state = openni::STATUS_OK;
	while (state == openni::STATUS_OK)
	{
		state = openni::OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);
		if (state == openni::STATUS_OK)
		{
			switch (streamReady)
			{
			case 0:
				fetchDepthFrame(depth);
				break;
			case 1:
				fetchRGBFrame(rgb);
				break;
			default:
				printf("Unexpected stream number!\n");
				return false;
			}
		}
	}

	if (!colorFrameRef || !depthFrameRef || !colorFrameRef->isValid() || !depthFrameRef->isValid())
		return false;

	return true;
}

void PrimeSense::fetchRGBFrame(cv::Mat& rgb)
{
	if (colorStream->readFrame(colorFrameRef) != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}

	rgb = cv::Mat(height, width, CV_8UC3, const_cast<void*>(colorFrameRef->getData()));
}

void PrimeSense::fetchDepthFrame(cv::Mat& depth)
{
	if (depthStream->readFrame(depthFrameRef) != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}

	depth = cv::Mat(height, width, CV_16UC1, const_cast<void*>(depthFrameRef->getData()));
}

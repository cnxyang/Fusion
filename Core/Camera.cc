#include "Camera.h"

camera::camera() :
		camera(640, 480, 30) {
}

camera::camera(int cols, int rows, int fps) :
		mCols(cols), mRows(rows), mFPS(fps), mpDevice(nullptr), mpColorStream(
				nullptr), mpColorFrame(nullptr), mpDepthStream(nullptr), mpDepthFrame(
				nullptr) {

	initCamera();
	startStreaming();
}

camera::~camera() {

	stopStreaming();
}

void camera::initCamera() {
	if (openni::OpenNI::initialize() != openni::STATUS_OK) {
		printf("OpenNI Initialisation Failed with Error Message : %s\n",
				openni::OpenNI::getExtendedError());
		exit(0);
	}

	mpDevice = new openni::Device();
	if (mpDevice->open(openni::ANY_DEVICE) != openni::STATUS_OK) {
		printf("Couldn't open device\n%s\n",
				openni::OpenNI::getExtendedError());
		exit(0);
	}

	mpDepthStream = new openni::VideoStream();
	mpColorStream = new openni::VideoStream();
	if (mpDepthStream->create(*mpDevice, openni::SENSOR_DEPTH)
			!= openni::STATUS_OK
			|| mpColorStream->create(*mpDevice, openni::SENSOR_COLOR)
					!= openni::STATUS_OK) {

		printf("Couldn't create streaming service\n%s\n",
				openni::OpenNI::getExtendedError());
		exit(0);
	}

	// change pixel format, resolution and FPS
	openni::VideoMode depth_video_mode = mpDepthStream->getVideoMode();
	depth_video_mode.setResolution(mCols, mRows);
	depth_video_mode.setFps(mFPS);
	depth_video_mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

	openni::VideoMode color_video_mode = mpColorStream->getVideoMode();
	color_video_mode.setResolution(mCols, mRows);
	color_video_mode.setFps(mFPS);
	color_video_mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	// save customised mode
	mpDepthStream->setVideoMode(depth_video_mode);
	mpColorStream->setVideoMode(color_video_mode);

	// Note: Doing image registration earlier than this seems to fail
	if (mpDevice->isImageRegistrationModeSupported(
			openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
		if (mpDevice->setImageRegistrationMode(
				openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) == openni::STATUS_OK)
			printf("Depth To Colour Image Registration Set Success\n");
		else
			printf("Depth To Colour Image Registration Set FAILED\n");
	} else {
		printf("Depth To Colour Image Registration is NOT Supported!!!\n");
	}

	printf("OpenNI Camera Initialisation Complete!\n");
}

void camera::startStreaming() {
	mpDepthStream->setMirroringEnabled(false);
	mpColorStream->setMirroringEnabled(false);

	if (mpDepthStream->start() != openni::STATUS_OK) {
		printf("Couldn't start depth streaming service\n%s\n",
				openni::OpenNI::getExtendedError());
		exit(0);
	}

	if (mpColorStream->start() != openni::STATUS_OK) {
		printf("Couldn't start rgb streaming service\n%s\n",
				openni::OpenNI::getExtendedError());
		exit(0);
	}

	mpDepthFrame = new openni::VideoFrameRef();
	mpColorFrame = new openni::VideoFrameRef();

	printf("OpenNI Camera Streaming Started!\n");
}

void camera::stopStreaming() {
	mpDepthStream->stop();
	mpColorStream->stop();
	mpColorStream->destroy();
	mpDepthStream->destroy();
	mpDevice->close();
	openni::OpenNI::shutdown();
}

bool camera::fetchFrame(cv::Mat& depth, cv::Mat& rgb) {

	openni::VideoStream * streams[] = { mpDepthStream, mpColorStream };
	int streamReady = -1;
	auto state = openni::STATUS_OK;
	while (state == openni::STATUS_OK) {

		state = openni::OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);
		if (state == openni::STATUS_OK) {
			switch (streamReady) {
			case 0:
				fetchDepthFrame(depth);
				break;
			case 1:
				fetchColorFrame(rgb);
				break;
			default:
				printf("Unexpected stream number!\n");
				return false;
			}
		}
	}

	if (!mpColorFrame || !mpDepthFrame || !mpColorFrame->isValid()
			|| !mpDepthFrame->isValid())
		return false;

	return true;
}

void camera::fetchColorFrame(cv::Mat& rgb) {
	if (mpColorStream->readFrame(mpColorFrame) != openni::STATUS_OK) {
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}
	rgb = cv::Mat(mRows, mCols, CV_8UC3,
			const_cast<void*>(mpColorFrame->getData()));
}

void camera::fetchDepthFrame(cv::Mat& depth) {
	if (mpDepthStream->readFrame(mpDepthFrame) != openni::STATUS_OK) {
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}
	depth = cv::Mat(mRows, mCols, CV_16UC1,
			const_cast<void*>(mpDepthFrame->getData()));
}

int camera::rows() const {
	return mRows;
}

int camera::fps() const {
	return mFPS;
}

int camera::cols() const {
	return mCols;
}

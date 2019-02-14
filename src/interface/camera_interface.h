#ifndef RGBD_CAMERA_INTERFACE
#define RGBD_CAMERA_INTERFACE

#include <iostream>
#include <opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <OpenNI.h>

class RGBDCameraInterface
{
public:

	RGBDCameraInterface();

	~RGBDCameraInterface();

	void start_camera_streaming();

	void stop_camera_streaming();

	/** ESSENTIAL: Read the next pair of images. return false if there is none */
	bool read_next_images(cv::Mat &image, cv::Mat &depth);

	/** MUTATOR: Return the time stamp of the current frame */
	double get_current_timestamp() const;

	/** MUTATOR: Return the id of the current frame */
	unsigned int get_current_id() const;

	void set_auto_exposure_enabled(bool enabled);

	void set_white_balance_enabled(bool enabled);

private:

	double time_stamp;
	unsigned int id;

	openni::Device *device;
	openni::VideoStream *rgb_stream, *depth_stream;
	openni::VideoFrameRef *rgb_frame, *depth_frame;
	openni::CameraSettings *settings;

	int image_width, image_height, frame_rate;
};

#endif

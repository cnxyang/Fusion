#include "RealSense.hpp"
#include <cstring>

RealSense::RealSense(int cols, int rows) :
		mcols(cols), mrows(rows) {
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_DEPTH, cols, rows, RS2_FORMAT_Z16, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, cols, rows, RS2_FORMAT_RGB8, 30);
	selection = pipe.start(cfg);

	auto profile = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto i = profile.get_intrinsics();
	fx = i.fx;
	fy = i.fy;
	cx = i.ppx;
	cy = i.ppy;
}

bool RealSense::Poll_for_frame(cv::Mat& imRGB, cv::Mat& imD) {
	if (pipe.poll_for_frames(&frames)) {
		color = frames.first(RS2_STREAM_COLOR);
		depth = frames.first(RS2_STREAM_DEPTH);

		imRGB = cv::Mat(mrows, mcols, CV_8UC3, const_cast<void*>(color.get_data()));
		imD = cv::Mat(mrows, mcols, CV_16UC1, const_cast<void*>(depth.get_data()));

		return true;
	} else {
		return false;
	}
}


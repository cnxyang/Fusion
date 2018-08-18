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

bool RealSense::Poll_for_frame(void*& rgb, void*& d) {
	if (pipe.poll_for_frames(&frames)) {
		color = frames.first(RS2_STREAM_COLOR);
		depth = frames.first(RS2_STREAM_DEPTH);

		std::memcpy((void*) rgb, color.get_data(),
				sizeof(unsigned char) * 3 * mcols * mrows);
		std::memcpy((void*) d, depth.get_data(),
				sizeof(ushort) * mcols * mrows);
		return true;
	} else {
		return false;
	}
}


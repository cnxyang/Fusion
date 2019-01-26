#ifndef __FRAME__
#define __FRAME__

#include <opencv2/opencv.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "slam_core/include/pose_graph_opt/keyframe_graph.h"

class FrameA
{
public:
	FrameA();

	Sophus::SE3d get_absolute_pose() const;
	cv::Mat get_image() const;
	cv::Mat get_depth() const;
	FrameA *get_parent() const;

private:
	FrameA *parent;
	cv::Mat image, depth;
	KeyFrameGraphA::Node node;
};

#endif

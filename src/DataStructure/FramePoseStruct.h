#ifndef FRAME_POSE_STRUCT__
#define FRAME_POSE_STRUCT__

#include <sophus/se3.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>

class Frame;

class FramePoseStruct
{
public:

	FramePoseStruct(Frame * frame);
	FramePoseStruct * trackingParent;
	Frame * frame;
	bool isRegisteredInGraph;
	bool isOptimised;
	bool isInGraph;
	g2o::VertexSE3Expmap * graphVertex;

	Sophus::SE3d thisToParent_raw;
	Sophus::SE3d & getCamToWorld();

private:

	Sophus::SE3d camToWorld;
};


#endif

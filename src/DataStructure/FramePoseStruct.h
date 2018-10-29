#ifndef FRAME_POSE_STRUCT__
#define FRAME_POSE_STRUCT__

#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Utilities/SophusUtil.h"

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

	SE3 thisToParent_raw;
	inline SE3 getCamToWorld();

private:

	SE3 camToWorld;
};

inline SE3 FramePoseStruct::getCamToWorld()
{
	return camToWorld;
}

#endif

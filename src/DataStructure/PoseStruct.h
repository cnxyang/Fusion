#pragma once
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Utilities/SophusUtil.h"

class Frame;

class PoseStruct
{
public:

	PoseStruct(Frame* frame);
	PoseStruct* parentPose;
	Frame* frame;
	bool isRegisteredInGraph;
	bool isOptimised;
	bool isInGraph;
	g2o::VertexSE3Expmap* graphVertex;

	SE3 thisToParent;
	inline SE3 getCamToWorld();

private:

	SE3 camToWorld;
};

inline SE3 PoseStruct::getCamToWorld()
{
	return camToWorld;
}

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
	SE3 camToWorld;
};

class DevicePoseStruct
{
	Matrix3f Rotation;
	Matrix3f InvRotation;
	float3 translation;
};

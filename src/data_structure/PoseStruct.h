#pragma once
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "SophusUtil.h"

class Frame;

class PoseStruct
{
public:

	PoseStruct(Frame* frame);
	void applyPoseUpdate();

	PoseStruct* parentPose;
	Frame* frame;
	bool isRegisteredInGraph;
	bool isOptimised;
	bool isInGraph;
	g2o::VertexSE3Expmap* graphVertex;

	SE3 thisToParent;
	SE3 camToWorld;
	float diff;
};

class DevicePoseStruct
{
	Matrix3f Rotation;
	Matrix3f InvRotation;
	float3 translation;
};

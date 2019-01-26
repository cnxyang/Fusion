#include "Frame.h"
#include "PoseStruct.h"

PoseStruct::PoseStruct(Frame* frame) :
	graphVertex(0), frame(frame), isOptimised(false),
	isInGraph(false), isRegisteredInGraph(false),
	parentPose(0), is_keyframe(false)
{
	camToWorld = thisToParent = SE3();
}

void PoseStruct::applyPoseUpdate()
{
	if(isOptimised && isInGraph && graphVertex)
	{
		camToWorld = QuattoSE3(graphVertex->estimate());
		this->diff = -1;
		isOptimised = false;
	}
}



//==============================================

Sophus::SE3d PoseStruct::get_absolute_pose() const
{
	if(is_keyframe)
		return camToWorld;
	else
		return parentPose->camToWorld * pose_from_parent;
}

#include "Frame.h"
#include "PoseStruct.h"

PoseStruct::PoseStruct(Frame* frame) :
	graphVertex(0), frame(frame), isOptimised(false),
	isInGraph(false), isRegisteredInGraph(false),
	parentPose(0)
{
	camToWorld = thisToParent = SE3();
}
#include "Frame.h"
#include "FramePoseStruct.h"

FramePoseStruct::FramePoseStruct(Frame* frame) :
	graphVertex(0), frame(frame), isOptimised(false),
	isInGraph(false), isRegisteredInGraph(false),
	parentPose(0)
{
	camToWorld = thisToParent = SE3();
}

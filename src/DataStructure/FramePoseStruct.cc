#include "Frame.h"
#include "FramePoseStruct.h"

FramePoseStruct::FramePoseStruct(Frame * frame) :
	graphVertex(0), frame(frame), isOptimised(false),
	isInGraph(false), isRegisteredInGraph(false),
	trackingParent(0)
{
	camToWorld = thisToParent_raw = Sophus::SE3d();
}

Sophus::SE3d & FramePoseStruct::getCamToWorld()
{
	return camToWorld;
}

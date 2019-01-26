#include "core/frame.h"

Sophus::SE3d FrameA::get_absolute_pose() const
{
	if(node.is_registered)
		return node.pose;
	else
		return parent->get_absolute_pose() * node.pose;
}

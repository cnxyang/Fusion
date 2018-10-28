#include "DataStructure/Frame.h"
#include "ICPTracking/PointCloud.h"

PointCloud::PointCloud(): memoryAllocated(false), frame(0)
{

}

void PointCloud::importData(Frame* frame, bool useRGB)
{
	if(memoryAllocated == false)
	{
		for(int level = 0; level < PYRAMID_LEVELS; ++level)
		{
			int width = frame->width(level);
			int height = frame->height(level);
			vmap[level].create(width, height);
			nmap[level].create(width, height);
			depth[level].create(width, height);
			image[level].create(width, height);
			dIdx[level].create(width, height);
			dIdy[level].create(width, height);

			if(level == 0)
			{
				image_raw.create(width, height);
				depth_raw.create(width, height);
			}
		}

		memoryAllocated = true;
	}

	depth_raw.upload(frame->data.depth.data, frame->data.depth.step);

	this->frame = frame;
}

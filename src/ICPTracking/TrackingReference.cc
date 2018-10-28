#include "DataStructure/Frame.h"
#include "TrackingReference.h"

TrackingReference::TrackingReference(int trackingLevel)
{

}

void TrackingReference::populateICPData(Frame* frame, bool useRGB)
{
	if(!cloud[0].memoryAllocated)
	{
		for(int level = 0; level < PYRAMID_LEVELS; ++level)
		{
			int width = frame->cols(level);
			int height = frame->rows(level);
			cloud[level].points.create(width, height);
			cloud[level].normal.create(width, height);
			cloud[level].image.create(width, height);
			cloud[level].depth.create(width, height);
			cloud[level].dIdx.create(width, height);
			cloud[level].dIdy.create(width, height);
			if(level == 0) {
				cloud[level].rawDepth.create(width, height);
				cloud[level].rawColor.create(width, height);
			}

			cloud[level].memoryAllocated = true;
		}
	}

	cloud[0].rawColor.upload(frame->data.image.data, frame->data.image.step);

	if(useRGB)
	{

	}

	cloud[0].rawDepth.upload(frame->data.depth.data, frame->data.depth.step);

	for(int level = 0; level < PYRAMID_LEVELS; ++level)
	{
		if(level == 0)
		{

		}
	}

	this->frame = frame;
}

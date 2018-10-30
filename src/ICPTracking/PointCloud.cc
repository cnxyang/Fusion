#include "DataStructure/Frame.h"
#include "ICPTracking/PointCloud.h"
#include "GPUWrapper/DeviceFuncs.h"

#define DEPTH_SCALE 1000.f
#define DEPTH_CUTOFF 3.0f

PointCloud::PointCloud(): memoryAllocated(false), frame(0)
{

}

void PointCloud::importFrame(Frame* frame, bool useRGB)
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

	DeviceArray2D<unsigned short> depthInput(frame->width(), frame->height());
	depthInput.upload(frame->data.depth.data, frame->data.depth.step);
	image_raw.upload(frame->data.image.data, frame->data.image.step);
	FilterDepth(depthInput, depth_raw, depth[0], DEPTH_SCALE, DEPTH_CUTOFF);
	ImageToIntensity(image_raw, image[0]);

	for(int i = 1; i < PYRAMID_LEVELS; ++i) {
		PyrDownGauss(depth[i - 1], depth[i]);
		PyrDownGauss(image[i - 1], image[i]);
	}

	for(int i = 0; i < PYRAMID_LEVELS; ++i) {
		float fx = frame->getfx(i);
		float fy = frame->getfy(i);
		float cx = frame->getcx(i);
		float cy = frame->getcy(i);
		ComputeVMap(depth[i], vmap[i], fx, fy, cx, cy, DEPTH_CUTOFF);
		ComputeNMap(vmap[i], nmap[i]);
		ComputeDerivativeImage(image[i], dIdx[i], dIdy[i]);
	}

	this->frame = frame;
}

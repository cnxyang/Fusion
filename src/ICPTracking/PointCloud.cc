#include "DataStructure/Frame.h"
#include "ICPTracking/PointCloud.h"
#include "GPUWrapper/DeviceFuncs.h"

#define DEPTH_SCALE 1000.f
#define DEPTH_CUTOFF 3.0f

PointCloud::PointCloud():
	memoryAllocated(false), frame(0)
{

}

PointCloud::~PointCloud()
{
	frame = 0;

	for(int level = 0; level < NUM_PYRS; ++level)
	{
		vmap[level].release();
		nmap[level].release();
		depth[level].release();
		image[level].release();
		dIdx[level].release();
		dIdy[level].release();
	}

	image_raw.release();
	depth_float.release();
}

void PointCloud::generateCloud(Frame* frame)
{
	if(!memoryAllocated)
	{
		for(int level = 0; level < NUM_PYRS; ++level)
		{
			int width = frame->width(level);
			int height = frame->height(level);
			std::cout << width << " " << height << std::endl;
			vmap[level].create(width, height);
			nmap[level].create(width, height);
			depth[level].create(width, height);
			image[level].create(width, height);
			dIdx[level].create(width, height);
			dIdy[level].create(width, height);

			if(level == 0)
			{
				image_raw.create(width, height);
				depth_float.create(width, height);
				depth_ushort.create(width, height);
			}
		}

		memoryAllocated = true;
	}

	// Upload raw depth onto GPU memory
	depth_ushort.upload(frame->data.depth.data, frame->data.depth.step);

	// Upload raw color onto GPU memory
	image_raw.upload(frame->data.image.data, frame->data.image.step);

	// Do a bilateral filtering before goes into tracking
	FilterDepth(depth_ushort, depth_float, depth[0], DEPTH_SCALE, DEPTH_CUTOFF);

	// Convert RGB images into gray-scale images
	ImageToIntensity(image_raw, image[0]);

	// Generate image pyramids
	for(int level = 1; level < NUM_PYRS; ++level) {
		PyrDownGauss(depth[level - 1], depth[level]);
		PyrDownGauss(image[level - 1], image[level]);
	}

	// Generate vertices and normals;
	for(int level = 0; level < NUM_PYRS; ++level)
	{
		float fx = frame->getfx(level);
		float fy = frame->getfy(level);
		float cx = frame->getcx(level);
		float cy = frame->getcy(level);

		// Vertices
		ComputeVMap(depth[level], vmap[level], fx, fy, cx, cy, DEPTH_CUTOFF);

		// Normals
		ComputeNMap(vmap[level], nmap[level]);

		// Compute derivative images ( sobel operation )
		ComputeDerivativeImage(image[level], dIdx[level], dIdy[level]);
	}

	// Update reference frame
	this->frame = frame;
}

void PointCloud::setReferenceFrame(Frame* frame)
{
	this->frame = frame;
}

void PointCloud::generatePyramid()
{
	for(int level = 1; level < NUM_PYRS; ++level) {
		ResizeMap(vmap[level - 1], nmap[level - 1], vmap[level], nmap[level]);
	}
}

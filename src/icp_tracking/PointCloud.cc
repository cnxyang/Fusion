#include "Frame.h"
#include "PointCloud.h"
#include "DeviceFuncs.h"
#include "Settings.h"

#define DEPTH_SCALE 5000.f
#define DEPTH_CUTOFF 10.0f

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

void PointCloud::generateCloud(Frame* frame, bool useRGB)
{
	if(!memoryAllocated)
	{
		for(int level = 0; level < NUM_PYRS; ++level)
		{
			int width = frame->width(level);
			int height = frame->height(level);
			vmap[level].create(width, height);
			nmap[level].create(width, height);
			depth[level].create(width, height);

			image[level].create(width, height);
			dIdx[level].create(width, height);
			dIdy[level].create(width, height);
			dZdx[level].create(width, height);
			dZdy[level].create(width, height);

			if(level == 0)
			{
				image_raw.create(width, height);
				depth_float.create(width, height);
				depth_ushort.create(width, height);
				weight.create(width, height);
			}
		}

		memoryAllocated = true;
	}

	// Upload raw depth onto GPU memory
	depth_ushort.upload(frame->data.depth.data, frame->data.depth.step);

	// Upload raw colour onto GPU memory
//	image_raw.upload(frame->data.image.data, frame->data.image.step);

	// Upload weight (if exists)
//	if(!frame->data.weight.empty())
//		weight.upload(frame->data.weight.data, frame->data.weight.step);
//	else
//		weight.clear();

	// Do a bilateral filtering before goes into tracking
	FilterDepth(depth_ushort, depth[0], depth_float,  DEPTH_SCALE, DEPTH_CUTOFF);

	// Convert RGB images into gray-scale images
//	if(useRGB)
//	{
//		ImageToIntensity(image_raw, image[0]);
//	}


	cv::Mat img;
	cv::cvtColor(frame->data.image, img, cv::COLOR_RGB2GRAY);
	img.convertTo(img, CV_32FC1);
	image[0].upload(img.data, img.step);

	// Generate image pyramids
	for(int level = 1; level < NUM_PYRS; ++level) {
		PyrDownGauss(depth[level - 1], depth[level]);
		if(useRGB)
		{
			pyrdown_image_mean(image[level - 1], image[level]);
		}
	}

	// Generate vertices and normals;
	for(int level = 0; level < NUM_PYRS; ++level)
	{
		float fx = frame->fx(level);
		float fy = frame->fy(level);
		float cx = frame->cx(level);
		float cy = frame->cy(level);

		// Vertices
		ComputeVMap(depth[level], vmap[level], fx, fy, cx, cy, DEPTH_CUTOFF);

		// Normals
		ComputeNMap(vmap[level], nmap[level]);

		if(useRGB)
		{
			// Compute derivative images ( sobel operation )
			compute_image_derivatives(image[level], dIdx[level], dIdy[level]);
		}
	}

	// Update reference frame
	this->frame = frame;

//	cv::Mat nn(480, 640, CV_32FC4);
//	nmap[0].download(nn.data, nn.step);
//	cv::imshow("nn", nn);
//	cv::waitKey(0);
}

void PointCloud::downloadFusedMap()
{
	cv::Mat img(frame->height(), frame->width(), CV_32FC1);
	depth_float.download(img.data, img.step);
	frame->data.depth.release();
	img.convertTo(frame->data.depth, CV_16UC1, 5000);
	img.release();
	frame->data.weight.create(frame->height(), frame->width(), CV_32SC1);
	weight.download(frame->data.weight.data, frame->data.weight.step);
}

void PointCloud::setReferenceFrame(Frame* frame)
{
	this->frame = frame;
}

#include "Frame.h"
#include "PoseStruct.h"

Frame::Frame(cv::Mat & image, cv::Mat & depth, int id, Eigen::Matrix3f K, double timeStamp) :
		poseStruct(0), keyPointStruct(0), point_struct(new PointStruct(image))
{
	initialize(image.cols, image.rows, id, K, timeStamp);

	image.copyTo(data.image);
	depth.copyTo(data.depth);

	point_struct->detect();
}

void Frame::initialize(int width, int height, int id, Eigen::Matrix3f & K, double timeStamp)
{
	data.id = id;

	poseStruct = new PoseStruct(this);

	data.K[0] = K;
	data.fx[0] = K(0, 0);
	data.fy[0] = K(1, 1);
	data.cx[0] = K(0, 2);
	data.cy[0] = K(1, 2);

	data.KInv[0] = data.K[0].inverse();
	data.fxInv[0] = data.KInv[0](0, 0);
	data.fyInv[0] = data.KInv[0](1, 1);
	data.cxInv[0] = data.KInv[0](0, 2);
	data.cyInv[0] = data.KInv[0](1, 2);

	data.timeStamp = timeStamp;

	for(int level = 0; level < NUM_PYRS; ++level)
	{
		data.width[level] = width >> level;
		data.height[level] = height >> level;

		if (level > 0)
		{
			data.fx[level] = data.fx[level-1] * 0.5;
			data.fy[level] = data.fy[level-1] * 0.5;
			data.cx[level] = (data.cx[0] + 0.5) / ((int) 1 << level) - 0.5;
			data.cy[level] = (data.cy[0] + 0.5) / ((int) 1 << level) - 0.5;

			data.K[level]  << data.fx[level], 0.0, data.cx[level],
							  0.0, data.fy[level], data.cy[level],
							  0.0, 0.0, 1.0;

			data.KInv[level] = (data.K[level]).inverse();
			data.fxInv[level] = data.KInv[level](0, 0);
			data.fyInv[level] = data.KInv[level](1, 1);
			data.cxInv[level] = data.KInv[level](0, 2);
			data.cyInv[level] = data.KInv[level](1, 2);
		}
	}
}

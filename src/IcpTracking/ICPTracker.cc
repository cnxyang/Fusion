#include "Frame.h"
#include "ICPTracker.h"
#include "DeviceFuncs.h"
#include "PointCloud.h"

ICPTracker::ICPTracker(int w, int h, Eigen::Matrix3f K) :
	icpInlierRatio(0), rgbInlierRatio(0), lastIcpError(0),
	lastRgbError(0), trackingWasGood(true)
{
	sumSE3.create(29, 96);
	outSE3.create(29);
	sumRES.create(2, 96);
	outRES.create(2);

	for(int level = 0; level < NUM_PYRS; ++level)
	{
		iterations[level] = 0;
	}
}

ICPTracker::~ICPTracker()
{
	sumSE3.release();
	outSE3.release();
	sumRES.release();
	outRES.release();
}

// return frame to reference pose transformation
SE3 ICPTracker::trackSE3(PointCloud* ref, PointCloud* target, SE3 estimate, bool useRGB)
{
	// return empty transform if data are not properly initiated
	// (normal won't happen)
	if(!ref->memoryAllocated || !target->memoryAllocated)
	{
		trackingWasGood = false;
		printf("Point Cloud NOT properly initiated!\n");
		return estimate;
	}

	// Temporary variable for storing residual and raw result
	float residual[2] = { 0.0f, 0.0f };

	// Total Reduction results
	Eigen::Matrix<double, 6, 6> matrixA;
	Eigen::Matrix<double, 6, 1> vectorb;

	// ICP reduction results
	Eigen::Matrix<double, 6, 6> matrixAicp;
	Eigen::Matrix<double, 6, 1> vectorbicp;

	// RGB reduction results
	Eigen::Matrix<double, 6, 6> matrixArgb;
	Eigen::Matrix<double, 6, 1> vectorbrgb;

	// se3
	Eigen::Matrix<double, 6, 1> result;

	// compatibility issue with old codes
	// TODO: get rid of this ASAP
	CameraIntrinsics K;
	K.fx = ref->frame->getfx(0);
	K.fy = ref->frame->getfy(0);
	K.cx = ref->frame->getcx(0);
	K.cy = ref->frame->getcy(0);

	// Store current frame poses
	SE3 refPose = SE3();
	SE3 frameToRef = estimate;
	SE3 framePose = refPose * frameToRef.inverse();

	for (int level = NUM_PYRS - 1; level >= 0; --level)
	{
		for (int iter = 0; iter < iterations[level]; ++iter)
		{
			ICPStep(target->vmap[level],
					ref->vmap[level],
					target->nmap[level],
					ref->nmap[level],
					SE3toMatrix3f(framePose),
					SE3toFloat3(framePose),
					K(level),
					sumSE3,
					outSE3,
					residual,
					matrixAicp.data(),
					vectorbicp.data());

			lastIcpError = sqrt(residual[0]) / (residual[1] < 1e-6 ? 1 : residual[1]);
			icpInlierRatio = residual[1] / target->frame->totalPixel(level);

			if(std::isnan(lastIcpError))
			{
				printf("Tracking FAILED with invalid ICP residual.\n");
				trackingWasGood = false;
				return estimate;
			}

			if(useRGB)
			{
				RGBStep(target->image[level],
						ref->image[level],
						target->vmap[level],
						ref->vmap[level],
						target->dIdx[level],
						target->dIdy[level],
						SE3toMatrix3f(framePose),
						SE3toMatrix3f(framePose.inverse()),
						SE3toMatrix3f(refPose),
						SE3toMatrix3f(refPose.inverse()),
						SE3toFloat3(framePose),
						SE3toFloat3(refPose),
						K(level),
						sumSE3,
						outSE3,
						sumRES,
						outRES,
						residual,
						matrixArgb.data(),
						vectorbrgb.data());

				lastRgbError = sqrt(residual[0]) / (residual[1] < 1e-6 ? 1 : residual[1]);
				rgbInlierRatio = residual[1] / target->frame->totalPixel(level);

				if(std::isnan(lastIcpError))
				{
					printf("Tracking FAILED with invalid RGB residual.\n");
					trackingWasGood = false;
					return estimate;
				}

				matrixA = RGBWeight * matrixArgb;
				matrixA += matrixAicp;
				vectorb = RGBWeight * vectorbrgb;
				vectorb += vectorbicp;
			}
			else
			{
				matrixA = matrixAicp;
				vectorb = vectorbicp;
			}

			result = matrixA.ldlt().solve(vectorb);
			auto frameToRefUpdate = SE3::exp(result);
			frameToRef = frameToRefUpdate * frameToRef;
			framePose = refPose * frameToRef.inverse();
		}
	}

	trackingWasGood = true;
	return frameToRef;
}

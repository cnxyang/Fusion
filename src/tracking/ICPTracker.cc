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
	// (normally it won't happen)
	if(!ref->memoryAllocated || !target->memoryAllocated)
	{
		trackingWasGood = false;
		printf("Point Cloud NOT properly initiated!\n");
		return estimate;
	}

	float residual[2] = { 0.0f, 0.0f };
	// Store current frame poses
	SE3 refPose = SE3();
	SE3 frameToRef = estimate;
	SE3 framePose = refPose * frameToRef.inverse();

	for (int level = NUM_PYRS - 1; level >= 0; --level)
	{
		DeviceArray2D<float4>& VMapNext = target->vmap[level];
		DeviceArray2D<float4>& NMapNext = target->nmap[level];
		DeviceArray2D<float4>& VMapLast = ref->vmap[level];
		DeviceArray2D<float4>& NMapLast = ref->nmap[level];

		float K[4] =
		{
				ref->frame->fx(level),
				ref->frame->fy(level),
				ref->frame->cx(level),
				ref->frame->cy(level),
		};

		for (int iter = 0; iter < iterations[level]; ++iter)
		{
			// Do ICP reduce sum
			ICPStep(VMapNext, NMapNext,
					VMapLast, NMapLast,
					sumSE3,	outSE3,
					SE3toMatrix3f(framePose),
					SE3toFloat3(framePose),
					K, residual,
					matrixAicp.data(),
					vectorbicp.data());

			int icpCount = (int)(residual[1] < 1e-6 ? 1 : residual[1]);
			lastIcpError = sqrt(residual[0]) / (float)icpCount;
			icpInlierRatio = (float)icpCount / target->frame->pixel(level);

			if(std::isnan(lastIcpError) || icpCount == 1)
			{
				printf("Tracking FAILED with invalid ICP Residual.\n");
				trackingWasGood = false;
				return SE3();
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
						K,
						sumSE3,
						outSE3,
						sumRES,
						outRES,
						residual,
						matrixArgb.data(),
						vectorbrgb.data());

				lastRgbError = sqrt(residual[0]) / (residual[1] < 1e-6 ? 1 : residual[1]);
				rgbInlierRatio = residual[1] / target->frame->pixel(level);

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

	Sophus::Vector3d diff = (estimate.inverse() * frameToRef).translation();
	if(diff.norm() > 0.1f)
		trackingWasGood = false;
	else
		trackingWasGood = true;
	return frameToRef;
}

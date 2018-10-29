#include "ICPTracking/ICPTracker.h"
#include "GPUWrapper/DeviceFuncs.h"
#include "ICPTracking/PointCloud.h"
#include "DataStructure/Frame.h"

ICPTracker::ICPTracker(int w, int h, Eigen::Matrix3f K) :
	trackingLevelBegin(2), trackingLevelEnd(0), icpInlierRatio(0),
	rgbInlierRatio(0), lastIcpError(0), lastRgbError(0),
	trackingWasGood(true), trackingReference(0), trackingTarget(0)
{
	trackingReference = new PointCloud();
	trackingTarget = new PointCloud();
	sumSE3.create(29, 96);
	outSE3.create(29);
	sumRES.create(2, 96);
	outRES.create(2);
}

ICPTracker::~ICPTracker()
{
	delete trackingReference;
	delete trackingTarget;
}

Sophus::SE3d ICPTracker::trackSE3(
		Frame * tracking_reference,
		Frame * tracking_target,
		SE3 initialEstimate,
		bool useRGB)
{
//	trackingReference->importData(tracking_reference, useRGB);
//
//	trackingTarget->importData(tracking_target, useRGB);
//
//	return trackSE3(trackingReference, trackingTarget, initialEstimate, useRGB);

	// Temporary variable for storing residual and raw result
	float residual[2] = { 0.0f, 0.0f };
	Eigen::Matrix<double, 6, 6> matrixA;
	Eigen::Matrix<double, 6, 1> vectorb;
	Eigen::Matrix<double, 6, 6> matrixAicp;
	Eigen::Matrix<double, 6, 1> vectorbicp;
	Eigen::Matrix<double, 6, 6> matrixArgb;
	Eigen::Matrix<double, 6, 1> vectorbrgb;
	Eigen::Matrix<double, 6, 1> result;

	// compatibility issue with old codes
	// TODO: get rid of this ASAP
	Intrinsics K;
	K.fx = Frame::fx(0);
	K.fy = Frame::fy(0);
	K.cx = Frame::cx(0);
	K.cy = Frame::cy(0);

	// store current frame poses
	SE3 refPose = tracking_reference->pose;
	SE3 frameToRef = initialEstimate;
	SE3 framePose = refPose * frameToRef.inverse();

	for (int level = trackingLevelBegin; level >= trackingLevelEnd; --level)
	{
		for (int iter = 0; iter < iterations[level]; ++iter)
		{
			ICPStep(tracking_target->vmap[level],
					tracking_reference->vmap[level],
					tracking_target->nmap[level],
					tracking_reference->nmap[level],
					SE3toMatrix3f(framePose),
					SE3toFloat3(framePose),
					SE3toMatrix3f(refPose),
					SE3toMatrix3f(refPose.inverse()),
					SE3toFloat3(refPose),
					K(level),
					sumSE3,
					outSE3,
					residual,
					matrixA.data(),
					vectorb.data());

			lastIcpError = sqrt(residual[0]) / (residual[1] < 1e-6 ? 1 : residual[1]);
			icpInlierRatio = residual[1] / (Frame::cols(level) * Frame::rows(level));

			if(std::isnan(lastIcpError))
			{
				printf("Tracking FAILED.\n");
				trackingWasGood = false;
				return SE3();
			}

			if(useRGB)
			{
				RGBStep(tracking_target->image[level],
						tracking_reference->image[level],
						tracking_target->vmap[level],
						tracking_reference->vmap[level],
						tracking_target->dIdx[level],
						tracking_target->dIdy[level],
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
				rgbInlierRatio = residual[1] / tracking_target->totalPixel(level);

				matrixA = 1e-4 * matrixArgb + matrixAicp;
				vectorb = 1e-4 * vectorbrgb + vectorbicp;
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

Sophus::SE3d ICPTracker::trackSE3(
		PointCloud * tracking_reference,
		Frame * tracking_target,
		SE3 initialEstimate,
		bool useRGB)
{
	trackingTarget->importData(tracking_target, useRGB);

	return trackSE3(tracking_reference, trackingTarget, initialEstimate, useRGB);
}

// return frame to reference pose transformation
SE3 ICPTracker::trackSE3(
	PointCloud * reference,
	PointCloud * target,
	SE3 initialEstimate,
	bool useRGB)
{
	// return empty transform if data are not properly initiated (normal won't happen)
	if(!reference->memoryAllocated || !target->memoryAllocated)
	{
		trackingWasGood = false;
		printf("Point Cloud NOT properly initiated!\n");
		return initialEstimate;
	}

	// Temporary variable for storing residual and raw result
	float residual[2] = { 0.0f, 0.0f };
	Eigen::Matrix<double, 6, 6> matrixA;
	Eigen::Matrix<double, 6, 1> vectorb;
	Eigen::Matrix<double, 6, 6> matrixAicp;
	Eigen::Matrix<double, 6, 1> vectorbicp;
	Eigen::Matrix<double, 6, 6> matrixArgb;
	Eigen::Matrix<double, 6, 1> vectorbrgb;
	Eigen::Matrix<double, 6, 1> result;

	// compatibility issue with old codes
	// TODO: get rid of this ASAP
	Intrinsics K;
	K.fx = reference->frame->getfx(0);
	K.fy = reference->frame->getfy(0);
	K.cx = reference->frame->getcx(0);
	K.cy = reference->frame->getcy(0);

	// store current frame poses
	SE3 refPose = reference->frame->pose;
	SE3 frameToRef = initialEstimate;
	SE3 framePose = refPose * frameToRef.inverse();

	for (int level = trackingLevelBegin; level >= trackingLevelEnd; --level)
	{
		for (int iter = 0; iter < iterations[level]; ++iter)
		{
			ICPStep(target->vmap[level],
					reference->vmap[level],
					target->nmap[level],
					reference->nmap[level],
					SE3toMatrix3f(framePose),
					SE3toFloat3(framePose),
					SE3toMatrix3f(refPose),
					SE3toMatrix3f(refPose.inverse()),
					SE3toFloat3(refPose),
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
				return initialEstimate;
			}

			if(useRGB)
			{
				RGBStep(target->image[level],
						reference->image[level],
						target->vmap[level],
						reference->vmap[level],
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
					return initialEstimate;
				}

				matrixA = 1e-4 * matrixArgb + matrixAicp;
				vectorb = 1e-4 * vectorbrgb + vectorbicp;
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

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
	sum.create(4, 96);
	out.create(4);
	corresp_image.create(640, 480);

	for(int level = 0; level < NUM_PYRS; ++level)
	{
		iterations[level] = 0;
	}
}

ICPTracker::~ICPTracker()
{

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
	SE3 framePose = estimate;
	Eigen::Matrix<double, 6, 1> last_result;

	trackingWasGood = true;

	for (int level = NUM_PYRS - 1; level >= 0; --level)
	{
		DeviceArray2D<float4>& VMapNext = target->vmap[level];
		DeviceArray2D<float4>& NMapNext = target->nmap[level];
		DeviceArray2D<float4>& VMapLast = ref->vmap[level];
		DeviceArray2D<float4>& NMapLast = ref->nmap[level];
		last_result.setOnes();
		float last_residual_error = std::numeric_limits<float>::max();

		float K[4] =
		{
				ref->frame->fx(level),
				ref->frame->fy(level),
				ref->frame->cx(level),
				ref->frame->cy(level),
		};

		DeviceArray<float> weight(VMapNext.cols * VMapNext.rows);
		DeviceArray<ResidualVector> residual_vec(weight.size);
		DeviceArray<Corresp> corresp_map(weight.size);
		float3 scale;


		for (int iter = 0; iter < iterations[0]; ++iter)
		{
			if(iter == 0)
				initialize_weight(weight);
			else
				compute_weight(residual_vec, weight, scale);

			compute_residual(VMapNext, VMapLast, NMapNext, NMapLast,
					target->image[level], ref->image[level], weight,
					residual_vec, corresp_map, SE3toMatrix3f(framePose),
					SE3toFloat3(framePose), K);

			Eigen::Matrix<float, 2, 2> sigma = compute_scale(residual_vec, weight, sum, out, VMapNext.cols * VMapNext.rows, point_ratio);
			scale = make_float3(sigma(0, 0), sigma(0, 1), sigma(1, 1));

			entropy = std::log(sigma.inverse().determinant());
			sumSE3.clear();
			outSE3.clear();
			compute_least_square(corresp_map, residual_vec, weight, VMapLast,
					VMapNext, NMapLast, ref->dIdx[level], ref->dIdy[level],
					sumSE3, outSE3, SE3toMatrix3f(framePose),
					SE3toMatrix3f(framePose.inverse()), SE3toFloat3(framePose),
					scale, K, matrixA.data(), vectorb.data(), residual);



//			compute_residual_sum(VMapNext, VMapLast, NMapNext, NMapLast,
//					target->image[level], ref->image[level],
//					SE3toMatrix3f(framePose), SE3toFloat3(framePose), K,
//					sum, out, sumSE3, outSE3, target->dIdx[level],
//					target->dIdy[level] , SE3toMatrix3f(framePose.inverse()),
//					residual, matrixA.data(), vectorb.data(), corresp_image);


			result = matrixA.ldlt().solve(vectorb);
			float diff = (result-last_result).norm();
			last_result = result;
			if(std::isnan(result(0)))
			{
//				trackingWasGood = false;
//				return estimate;
				std::cout << matrixA << std::endl;
			}

			float residual_error = sqrt(residual[0]) / residual[1];
			last_residual_error = residual_error;

			framePose = Sophus::SE3d::exp(result) * framePose;

			if(diff < 0.001)
				break;
		}
	}


//	float K[4] =
//	{
//			ref->frame->fx(0),
//			ref->frame->fy(0),
//			ref->frame->cx(0),
//			ref->frame->cy(0),
//	};
//	compute_residual_transformed(target->vmap[0], ref->vmap[0],
//			ref->nmap[0], target->image[0], ref->image[0], K,
//			SE3toMatrix3f(framePose), SE3toFloat3(framePose));
//	Sophus::Vector3d diff = (estimate.inverse() * frameToRef).translation();
//	if(diff.norm() > 0.3f)
//		trackingWasGood = false;
//	else
		trackingWasGood = true;
	return framePose;
}

#include <chrono>
#include <iostream>
#include <cmath>

#include "Timer.h"
#include "Solver.h"
#include "DeviceMap.h"
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;

bool Solver::SolveAbsoluteOrientation(vector<Vector3d>& src,
		vector<Vector3d>& ref, vector<bool>& outlier,
		Matrix4d& Td, int maxIter) {

	assert(src.size() == ref.size());

	Matrix3d R_best = Matrix3d::Identity();
	Vector3d t_best = Vector3d::Zero();
	int inliers_best = 0;
	int nMatches = src.size();

	auto now = chrono::system_clock::now();
	int seed = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();
	srand(seed);

	int nIter = 0;
	int nBadSamples = 0;
	float ratio = 0.0f;
	float confidence = 0.0f;
	float thresh_inlier = 0.05f;
	const float thresh_confidence = 0.95f;
	const int minIter = 20;
	if(nMatches < 3)
		return false;

	while (nIter < maxIter) {

		bool badSample = false;
		vector<int> samples;
		for (int i = 0; i < 3; ++i) {
			int s = rand() % nMatches;
			samples.push_back(s);
		}

		if (samples[0] == samples[1] ||
			samples[1] == samples[2] ||
			samples[2] == samples[0])
			badSample = true;

		Vector3d src_a = src[samples[0]];
		Vector3d src_b = src[samples[1]];
		Vector3d src_c = src[samples[2]];

		Vector3d ref_a = ref[samples[0]];
		Vector3d ref_b = ref[samples[1]];
		Vector3d ref_c = ref[samples[2]];

		float src_d = (src_b - src_a).cross(src_a - src_c).norm();
		float ref_d = (ref_b - ref_a).cross(ref_a - ref_c).norm();

		if (badSample || src_d < 1e-6 || ref_d < 1e-6) {
			nBadSamples++;
			nIter++;
			continue;
		}

		Vector3d src_mean = (src_a + src_b + src_c) / 3;
		Vector3d ref_mean = (ref_a + ref_b + ref_c) / 3;

		src_a -= src_mean;
		src_b -= src_mean;
		src_c -= src_mean;

		ref_a -= ref_mean;
		ref_b -= ref_mean;
		ref_c -= ref_mean;

		Matrix3d Ab = Matrix3d::Zero();
		Ab += src_a * ref_a.transpose();
		Ab += src_b * ref_b.transpose();
		Ab += src_c * ref_c.transpose();

		JacobiSVD<Matrix3d> svd(Ab, ComputeFullU | ComputeFullV);
		Matrix3d V = svd.matrixV();
		Matrix3d U = svd.matrixU();
		Matrix3d R = (V * U.transpose()).transpose();
		if(R.determinant() < 0)
			continue;
		Vector3d t = src_mean - R * ref_mean;

		int nInliers = 0;
		outlier.resize(src.size());
		fill(outlier.begin(), outlier.end(), true);
		for (int i = 0; i < src.size(); ++i) {
			double d = (src[i] - (R * ref[i] + t)).norm();
			if (d <= thresh_inlier) {
				nInliers++;
				outlier[i] = false;
			}
		}

		if (nInliers > inliers_best) {

			Ab = Matrix3d::Zero();
			src_mean = Vector3d::Zero();
			ref_mean = Vector3d::Zero();
			for (int i = 0; i < outlier.size(); ++i) {
				if (!outlier[i]) {
					src_mean += src[i];
					ref_mean += ref[i];
				}
			}

			src_mean /= nInliers;
			ref_mean /= nInliers;

			for (int i = 0; i < outlier.size(); ++i) {
				if (!outlier[i]) {
					Ab += (src[i] - src_mean) * (ref[i] - ref_mean).transpose();
				}
			}

			svd.compute(Ab, ComputeFullU | ComputeFullV);
			V = svd.matrixV();
			U = svd.matrixU();
			R_best = (V * U.transpose()).transpose();
			t_best = src_mean - R_best * ref_mean;
			inliers_best = nInliers;

			ratio = (float) nInliers / src.size();

			confidence = 1 - pow((1 - pow(ratio, 3)), nIter + 1);

			if (nIter >= minIter && confidence >= thresh_confidence)
				break;
		}

		nIter++;
	}

	Td.topLeftCorner(3, 3) = R_best;
	Td.topRightCorner(3, 1) = t_best;

//	if (confidence < 0.95) {
//		std::cout << "low confident" << std::endl;
//		Eigen::Vector3d angles = R_best.eulerAngles(0, 1, 2).array().sin();
//		if(angles.norm() >= 0.2 || t_best.norm() >= 0.1)
//			return false;
//	}

	return true;
}

float Solver::SolveICP(Frame& src, Frame& ref) {

//	float cost = 0;
//	const float w = 0.1;
//	const int iter[3] = { 10, 5, 3 };
//
//	Vector6d result;
//	Matrix6f host_a;
//	Vector6f host_b;
//
//	for (int i = 2; i >= 0; --i) {
//		for (int j = 0; j < iter[i]; j++) {
//
//			cost = ICPReduceSum(src, ref, i, host_a.data(), host_b.data());
//
//			Matrix6d dA_icp = host_a.cast<double>();
//			Vector6d db_icp = host_b.cast<double>();
//
////			cost = RGBReduceSum(src, ref, i, host_a.data(), host_b.data());
////			Matrix6d dA_rgb = host_a.cast<double>();
////			Vector6d db_rgb = host_b.cast<double>();
////			Matrix6d dA = w * w * dA_icp + dA_rgb;
////			Vector6d db = w * db_icp + db_rgb;
//
//			Matrix6d dA = dA_icp;
//			Vector6d db = db_icp;
//			result = dA.ldlt().solve(db);
//			auto e = Sophus::SE3d::exp(result);
//			auto dT = e.matrix();
//
//			Eigen::Matrix4d Tc = src.mPose;
//			Eigen::Matrix4d Tp = ref.mPose;
//			Tc = Tp * (dT.inverse() * Tc.inverse() * Tp).inverse();
//			src.SetPose(Tc);
//		}
//	}
//
//	return cost;
}

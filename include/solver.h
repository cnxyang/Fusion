#ifndef SOLVER_HPP__
#define SOLVER_HPP__

#include "frame.h"

#include <vector>
#include <Eigen/Dense>

class Solver {
public:
	static bool SolveAbsoluteOrientation(std::vector<Eigen::Vector3d>& src,
		std::vector<Eigen::Vector3d>& ref, std::vector<bool>& outliers,
		Eigen::Matrix4d& T, int maxIter);



static float SolveICP(Frame& src, Frame& ref);
};

static cv::Ptr<cv::cuda::DescriptorMatcher> key_matcher
= cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
static bool solveAbsoluteOrientation(std::vector<Eigen::Vector3d> & src_points,
									 std::vector<Eigen::Vector3d> & ref_points,
									 cv::cuda::GpuMat & src_descriptors,
									 cv::cuda::GpuMat & ref_descriptors,
									 std::vector<bool> & outlier_list,
									 Eigen::Matrix4d & delta_t,
									 int max_iterations);

#endif

#ifndef SOLVER_HPP__
#define SOLVER_HPP__

#include <vector>
#include "Frame.hpp"
#include <Eigen/Dense>

class Solver {
public:
	static bool SolveAbsoluteOrientation(std::vector<Eigen::Vector3d>& src, std::vector<Eigen::Vector3d>& ref, std::vector<bool>& outliers, Eigen::Matrix4d& T, int maxIter);
	static float SolveICP(Frame& src, Frame& ref);
};

#endif

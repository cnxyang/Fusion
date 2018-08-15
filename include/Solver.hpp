#ifndef SOLVER_HPP__
#define SOLVER_HPP__

#include <vector>
#include "Frame.hpp"
#include <Eigen/Dense>

using namespace std;

class Solver {
public:
	static bool SolveAbsoluteOrientation(vector<Eigen::Vector3d>& src, vector<Eigen::Vector3d>& ref, vector<bool>& outliers, Eigen::Matrix4d& T);
	static bool SolveICP(Frame& src, Frame& ref, Eigen::Matrix4d& Td);
};

#endif

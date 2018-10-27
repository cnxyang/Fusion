#pragma once

#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

struct KFConstraint
{
	KFConstraint() :
		first(0), second(0), edge(0)
	{
		informationMatrix.setZero();
	}

	Frame* first;
	Frame* second;
	g2o::SE3Quat firstToSecond;
	g2o::EdgeSE3Expmap* edge;
	Eigen::Matrix<double, 6, 6> informationMatrix;
};

class KeyFrameGraph
{
public:

	KeyFrameGraph();

	void addKeyFrame(Frame* frame);

	void insertConstraint(KFConstraint* constraint);

private:

	g2o::SparseOptimizer graph;

	std::vector<Frame *> keyFramesAll;
};

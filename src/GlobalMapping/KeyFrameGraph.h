#ifndef KEY_FRAME_GRAPH__
#define KEY_FRAME_GRAPH__

#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "DataStructure/Frame.h"

class Frame;

struct KFConstraint
{
	KFConstraint() :
		first(0), second(0), edge(0)
	{
		information.setZero();
	}

	Frame* first;
	Frame* second;
	g2o::SE3Quat firstToSecond;
	g2o::EdgeSE3Expmap* edge;
	Eigen::Matrix<double, 6, 6> information;
};

class KeyFrameGraph
{
public:

	KeyFrameGraph();

	void addKeyFrame(Frame* frame);

	void addFrame(Frame* frame);

	void insertConstraint(KFConstraint* constraint);

private:

	g2o::SparseOptimizer graph;

	std::vector<Frame *> keyFramesAll;
};

#endif

#pragma once
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "DataStructure/Frame.h"
#include <mutex>
#include <unordered_set>

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

struct TrackableKFStruct
{

};

class KeyFrameGraph
{
public:

	KeyFrameGraph(int w, int h, Eigen::Matrix3f K);

	~KeyFrameGraph();

	void addKeyFrame(Frame* frame);

	void addFrame(Frame* frame);

	void insertConstraint(KFConstraint* constraint);

	std::unordered_set<Frame*, std::hash<Frame*>> searchCandidates(Frame* kf);

	std::vector<TrackableKFStruct> findEuclideanOverlapFrames(Frame* kf, float distTH, float angleTH);

	inline std::vector<Frame*> getAllKeyFrames() const;

	std::vector<SE3> getAllKeyFramePoses() const;

private:

	g2o::SparseOptimizer graph;

	std::vector<Frame *> keyFramesAll;

	std::mutex keyFramesAllMutex;

	float fowX, fowY;
};

inline std::vector<Frame*> KeyFrameGraph::getAllKeyFrames() const
{
	return keyFramesAll;
}

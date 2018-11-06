#pragma once
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "DataStructure/Frame.h"
#include <mutex>
#include <unordered_set>

struct KFConstraintStruct
{
	KFConstraintStruct() :
		first(0), second(0), edge(0),
		idxInAllEdges(0)
	{
		information.setZero();
	}

	Frame* first;
	Frame* second;
	unsigned int idxInAllEdges;
	g2o::SE3Quat firstToSecond;
	g2o::EdgeSE3Expmap* edge;
	Eigen::Matrix<double, 6, 6> information;
};

struct TrackableKFStruct
{
	Frame* frame;
	SE3 ref2Frame;
	float dist;
	float angle;
};

class KeyFrameGraph
{
public:

	KeyFrameGraph(int w, int h, Eigen::Matrix3f K);
	KeyFrameGraph(const KeyFrameGraph&) = delete;
	KeyFrameGraph& operator=(const KeyFrameGraph&) = delete;
	~KeyFrameGraph();

	void addKeyFrame(Frame* frame);
	void addFrame(Frame* frame);
	void insertConstraint(KFConstraintStruct* constraint);
	int optimize(int iterations);
	bool addElementsFromBuffer();
	void updatePoseGraph();

	std::unordered_set<Frame*, std::hash<Frame*>> findTrackableCandidates(Frame* keyFrame);
	std::vector<SE3> keyframePoseAll() const;
	inline std::vector<Frame *> getKeyFramesAll() const;
	inline void reinitialiseGraph();

private:

	std::vector<TrackableKFStruct> findOverlappingFrames(Frame* frame, float distTH, float angleTH);

	g2o::SparseOptimizer graph;
	std::vector<Frame*> keyframesAll;
	std::mutex keyframesAllMutex;

	std::vector<Frame*> newKeyframesBuffer;
	std::mutex newKeyFrameMutex;

	std::vector<KFConstraintStruct*> edgesAll;
	std::vector<g2o::EdgeSE3Expmap*> newEdgeBuffer;
	std::mutex edgesListsMutex;

	std::mutex graphAccessMutex;

	bool hasUnupdatedPose;
	int nextEdgeId;
	float fowX, fowY;
};

inline std::vector<Frame*> KeyFrameGraph::getKeyFramesAll() const
{
	return keyframesAll;
}

inline void KeyFrameGraph::reinitialiseGraph()
{
	keyframesAll.clear();
	graph.clear();
	graph.clearParameters();
}

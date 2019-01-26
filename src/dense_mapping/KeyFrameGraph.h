#ifndef __KEY_FRAME_GRAPH__
#define __KEY_FRAME_GRAPH__

#include "Frame.h"
#include "EigenUtils.h"
#include <mutex>
#include <unordered_set>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

struct ConstraintStruct
{
	ConstraintStruct() :
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

	KeyFrameGraph(int width, int height, Eigen::Matrix3f K);
	~KeyFrameGraph();

	void optimizeGraph(int iterations);
	void addKeyFrame(Frame * keyFrame);
	void insertConstraint(ConstraintStruct * constraint);
	bool addElementsFromBuffer();
	void updatePoseGraph();

	std::unordered_set<Frame*, std::hash<Frame*>> findTrackableCandidates(Frame * keyFrame);
	std::vector<Sophus::SE3d> keyframePoseAll() const;
	std::vector<Frame *> getKeyFramesAll() const;
	void clearGraph();

private:

	bool hasUnupdatedPose;
	int nextEdgeId;
	float fowX, fowY;

	std::vector<TrackableKFStruct> findOverlappingFrames(Frame* frame, float distTH, float angleTH);
	g2o::SparseOptimizer graph;

	std::vector<Frame *> keyframesAll;
	std::vector<PoseStruct *> framePoseAll;
	std::vector<Frame *> newKeyframesBuffer;
	std::vector<ConstraintStruct *> edgesAll;
	std::vector<g2o::EdgeSE3Expmap *> newEdgeBuffer;

	std::mutex edgesListsMutex;
	std::mutex newKeyFrameMutex;
	std::mutex graphAccessMutex;
	std::mutex keyframesAllMutex;

public:

	void insert_frame_pose(Frame* frame);
	std::list<PoseStruct*> frames;
};

#endif

#ifndef __KEYFRAMEGRAPH__
#define __KEYFRAMEGRAPH__

#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Frame.h"
#include "EigenUtils.h"

class KeyFrameGraphA
{
public:
	KeyFrameGraphA();

	typedef struct Node
	{
		Frame *base;
		unsigned int node_id;
		Sophus::SE3d pose;
		Sophus::SE3d pose_opt;
		bool is_registered;
		g2o::VertexSE3Expmap *graphVertex;
	} *lpNode;

	typedef struct Edge
	{
		unsigned int edge_id;
		Node *first_node, *second_node;
		Sophus::SE3d pose_first_second;
		g2o::EdgeSE3Expmap *edge_g2o;
		Matrix6x6 information_matrix;
	} *lpEdge;

	void insert_key_frame(Frame* keyframe);
	void insert_frame(Frame* frame);
	void insert_edge(Edge* constraint);
	void batch_insert_edge(std::list<Edge*>& edge_list);
	void optimize(int total_iteration);
	void optimize_locally(int total_iteration);
	void optimize_globally(int total_iteration);
	void reinitialize_graph();

	std::list<Frame*> find_trackable_keyframes(Frame* keyframe) const;

	g2o::SparseOptimizer graph;
	std::list<Frame*> keyframe_list;
	std::queue<Frame*> local_map;
	std::list<PoseStruct*> frame_list;

	std::mutex pose_graph_mutex;
};

#endif

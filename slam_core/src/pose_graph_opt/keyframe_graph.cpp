#include "pose_graph_opt/keyframe_graph.h"


KeyFrameGraphA::KeyFrameGraphA()
{
	graph.setVerbose(false);
	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
	graph.setAlgorithm(solver);
}

void KeyFrameGraphA::insert_key_frame(Frame* keyframe)
{

}

void KeyFrameGraphA::optimize_locally(int total_iteration)
{
	while(local_map.size() > local_map_maxsize)
	{
		local_map.front()->poseStruct->graphVertex->setFixed(true);
		local_map.pop();
	}

	optimize(total_iteration);
}

void KeyFrameGraphA::optimize_globally(int total_iteration)
{
	auto iter = keyframe_list.begin();
	auto lend = keyframe_list.end();
	for(; iter != lend; ++iter)
	{
		auto keyframe = (*iter);
		if(keyframe->id() == 0)
			keyframe->poseStruct->graphVertex->setFixed(true);
		else
			keyframe->poseStruct->graphVertex->setFixed(false);
	}

	optimize(total_iteration);
}

void KeyFrameGraphA::optimize(int total_iteration)
{
	if (graph.edges().size() == 0)
		return;

	graph.initializeOptimization();
	graph.optimize(total_iteration);
}

void KeyFrameGraphA::reinitialize_graph()
{
	std::unique_lock<std::mutex> lock(pose_graph_mutex);
	graph.clear();
	graph.clearParameters();
	keyframe_list.clear();
	frame_list.clear();
}

void KeyFrameGraphA::insert_edge(Edge* constraint)
{

}

void KeyFrameGraphA::batch_insert_edge(std::list<Edge*>& edge_list)
{

}

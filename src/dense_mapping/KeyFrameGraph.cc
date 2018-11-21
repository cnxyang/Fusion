#include "KeyFrameGraph.h"
#include "Settings.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

KeyFrameGraph::KeyFrameGraph(int w, int h, Eigen::Matrix3f K) :
	nextEdgeId(0), hasUnupdatedPose(false)
{
	fowX = 2 * atanf((float)((w / K(0,0)) / 2.0f));
	fowY = 2 * atanf((float)((h / K(1,1)) / 2.0f));

	graph.setVerbose(false);
	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
	graph.setAlgorithm(solver);
}

KeyFrameGraph::~KeyFrameGraph()
{

}

void KeyFrameGraph::addKeyFrame(Frame* frame)
{
	if(frame->poseStruct->graphVertex != 0)	return;

	keyframesAll.push_back(frame);
	g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
	vertex->setId(frame->id());
	if(!frame->hasTrackingParent())
		vertex->setFixed(true);

	vertex->setEstimate(SE3toQuat(frame->pose()));
	vertex->setMarginalized(false);
	frame->poseStruct->graphVertex = vertex;
	newKeyframesBuffer.push_back(frame);
}

std::vector<SE3> KeyFrameGraph::keyframePoseAll() const
{
	std::vector<SE3> poses;
	std::transform(keyframesAll.begin(), keyframesAll.end(), std::back_inserter(poses), [](Frame* f) { return f->pose(); });
	return poses;
}

std::vector<TrackableKFStruct> KeyFrameGraph::findOverlappingFrames(Frame* frame, float distTH, float angleTH)
{
	float cosAngleTH = cosf(angleTH * 0.5f * (fowX + fowY));
	Eigen::Vector3d pos = frame->pose().translation();
	Eigen::Vector3d viewingDir = frame->pose().rotationMatrix().rightCols<1>();

	std::vector<TrackableKFStruct> potentialReferenceFrames;
	keyframesAllMutex.lock();
	for(unsigned int i = 0; i < keyframesAll.size(); ++i)
	{
		Eigen::Vector3d otherPos = keyframesAll[i]->pose().translation();
		Eigen::Vector3d dist = pos - otherPos;
		float dNorm2 = dist.dot(dist);
		if(dNorm2 > distTH)
			continue;

		Eigen::Vector3d otherViewingDir = keyframesAll[i]->pose().rotationMatrix().rightCols<1>();
		float dirDotProd = otherViewingDir.dot(viewingDir);
		if(dirDotProd < cosAngleTH)
			continue;

		potentialReferenceFrames.push_back(TrackableKFStruct());
		potentialReferenceFrames.back().frame = keyframesAll[i];
		potentialReferenceFrames.back().ref2Frame = keyframesAll[i]->pose().inverse() * frame->pose();
		potentialReferenceFrames.back().dist = dNorm2;
		potentialReferenceFrames.back().angle = dirDotProd;
	}

	keyframesAllMutex.unlock();
	return potentialReferenceFrames;
}

std::unordered_set<Frame*, std::hash<Frame*>> KeyFrameGraph::findTrackableCandidates(Frame* keyFrame)
{
	std::unordered_set<Frame*, std::hash<Frame*>> results;
	std::vector<TrackableKFStruct> potentialReferenceFrames = findOverlappingFrames(keyFrame, 0.2f, 0.4f);
	for(unsigned int i = 0; i < potentialReferenceFrames.size(); ++i)
		results.insert(potentialReferenceFrames[i].frame);
	return results;
}

void KeyFrameGraph::insertConstraint(KFConstraintStruct* constraint)
{
	g2o::EdgeSE3Expmap* edge = new g2o::EdgeSE3Expmap();
	edge->setId(nextEdgeId++);

	edge->setMeasurement(constraint->firstToSecond);
	edge->setVertex(0, constraint->first->poseStruct->graphVertex);
	edge->setVertex(1, constraint->second->poseStruct->graphVertex);
	edge->setInformation(constraint->information);

	constraint->edge = edge;
	newEdgeBuffer.push_back(edge);

	edgesListsMutex.lock();
	constraint->idxInAllEdges = edgesAll.size();
	edgesAll.push_back(constraint);
	edgesListsMutex.unlock();
}

bool KeyFrameGraph::addElementsFromBuffer()
{
	std::unique_lock<std::mutex> lock(graphAccessMutex);
	bool added = false;
	for(auto newKF : newKeyframesBuffer)
	{
		graph.addVertex(newKF->poseStruct->graphVertex);
		newKF->poseStruct->isInGraph = true;
		added = true;
	}

	newKeyframesBuffer.clear();
	for(auto edge : newEdgeBuffer)
	{
		graph.addEdge(edge);
		added = true;
	}

	newEdgeBuffer.clear();
	return added;
}

void KeyFrameGraph::updatePoseGraph()
{
	if(hasUnupdatedPose)
	{
		std::unique_lock<std::mutex> lock(graphAccessMutex);
		for (auto frame : keyframesAll)
		{
			frame->poseStruct->applyPoseUpdate();
		}
	}
}

int KeyFrameGraph::optimize(int iterations)
{
	if (graph.edges().size() == 0)
		return 0;

	graph.setVerbose(true);
	graph.initializeOptimization();

	hasUnupdatedPose = true;
	return graph.optimize(iterations, false);
}

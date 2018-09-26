#include "Optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

Optimizer::Optimizer() :
		map(NULL), noKeyFrames(0) {

}

void Optimizer::run() {

	while(1) {

		if(map->HasNewKF()) {

		}

		std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	}
}

void Optimizer::LocalBA() {

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
	g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
	);

	optimizer.setAlgorithm(solver);

	std::set<const KeyFrame *>::iterator iter;
	std::set<const KeyFrame *>::iterator lend;

	iter = map->localMap.begin();
	lend = map->localMap.end();
	for (; iter != lend; ++iter) {

		const KeyFrame * kf = *iter;
		g2o::SE3Quat pose(kf->Rotation().cast<double>(), kf->Translation().cast<double>());
		g2o::VertexSE3Expmap * vse3 = new g2o::VertexSE3Expmap();

		vse3->setId(kf->frameId);
		vse3->setFixed(false);
		vse3->setEstimate(pose);

		optimizer.addVertex(vse3);
	}

	optimizer.initializeOptimization();
	optimizer.setVerbose(true);
	optimizer.optimize(10);
}

void Optimizer::GlobalBA() {

}

void Optimizer::SetMap(Mapping * map_) {

	map = map_;
}

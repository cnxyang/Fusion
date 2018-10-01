#include "Optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

Optimizer::Optimizer() :
		map(NULL), noKeyFrames(0) {

}

void Optimizer::run() {

	while(1) {

		if(map->HasNewKF()) {

			localMap = map->LocalMap();

			if(localMap.size() > 5)
				LocalBA();

			map->hasNewKFFlag = false;
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
	int offset = localMap.size();

	for (int i = 0; i < offset; ++i) {
		KeyFrame * kf = localMap[i];
		g2o::SE3Quat pose(kf->Rotation().cast<double>(), kf->Translation().cast<double>());
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setId(i);
		vSE3->setEstimate(pose.inverse());
		if(i == 0)
			vSE3->setFixed(true);
		else
			vSE3->setFixed(false);

		optimizer.addVertex(vSE3);

		for (int j = 0; j < kf->N; ++j) {
			if (!kf->outliers[j] && kf->keyIndex[j] > -1) {

				g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSE3));
				e->Xw = kf->mapPoints[j].cast<double>();
				e->information() = Eigen::Matrix2d::Identity();
				Eigen::Vector2d obs = Eigen::Vector2d::Identity();
				obs << kf->keyPoints[j].pt.x, kf->keyPoints[j].pt.y;
				e->setMeasurement(obs);
				g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);

				e->fx = Frame::fx(0);
				e->fy = Frame::fy(0);
				e->cx = Frame::cx(0);
				e->cy = Frame::cy(0);

				optimizer.addEdge(e);
			}
		}
	}

	optimizer.initializeOptimization();
	optimizer.setVerbose(false);
	optimizer.optimize(10);

	for(int i = 1; i < localMap.size(); ++i) {
		 g2o::VertexSE3Expmap * vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
		 g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		 Eigen::Matrix4d eigMat = SE3quat_recov.to_homogeneous_matrix();
		 localMap[i]->newPose = eigMat.inverse().cast<float>();
	}
}

void Optimizer::GlobalBA() {

}

void Optimizer::SetMap(Mapping * map_) {

	map = map_;
}

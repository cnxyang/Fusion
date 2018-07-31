#include "Tracking.h"
#include "DeviceFunc.h"
#include "DeviceStruct.h"
#include "eigen3/Eigen/Dense"
#include "sophus/se3.hpp"
#include "Converter.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"


Tracking::Tracking() {
	mpMap = nullptr;
	mNextState = NOT_INITIALISED;
	mORBMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
}

bool Tracking::GrabImageRGBD(cv::Mat& imRGB, cv::Mat& imD) {

	mNextFrame = Frame(imRGB, imD);

	bool bOK = Track();
	if(!bOK)
		return false;

	NeedNewKeyFrame();

	if(mbNeedNewKF)
		CreateKeyFrame();

	mLastFrame = Frame(mNextFrame);

	return true;
}

bool Tracking::Track() {
	bool bOK;
	switch(mNextState) {
	case NOT_INITIALISED:
		bOK = InitTracking();
		break;

	case OK:
		bOK = TrackLastFrame();
		break;

	case LOST:
		bOK = Relocalisation();
		break;
	}

	if(!bOK) {
		mNextState = LOST;
//		Track();
	}
	else {
		mNextState = OK;
	}

	return bOK;
}

bool Tracking::InitTracking() {
	mpMap->SetFirstFrame(mNextFrame);
	mbNeedNewKF = true;
	mNextState = OK;
	mNoFrames = 0;
	return true;
}

bool Tracking::TrackLastFrame() {
	mNextFrame.SetPose(mLastFrame);
//	bool bOK = TrackMap();
	bool bOK = TrackFrame();
//	if(!bOK)
//		return false;
	TrackICP();
	return true;
}

bool Tracking::Relocalisation() {
	bool result = TrackMap();
	if(result)
		mNextState = OK;
	return result;
}

void Tracking::NeedNewKeyFrame() {
	if(mbNeedNewKF)
		return;

	Eigen::Vector3d p, q;
	p << mNextFrame.mtcw.at<float>(0), mNextFrame.mtcw.at<float>(1), mNextFrame.mtcw.at<float>(2);
	q << mLastKeyFrame.mtcw.at<float>(0), mLastKeyFrame.mtcw.at<float>(1), mLastKeyFrame.mtcw.at<float>(2);

	if((p - q).norm() > 0.5)
		mbNeedNewKF = true;
}

void Tracking::CreateKeyFrame() {
	mLastKeyFrame = KeyFrame(mNextFrame);
	mpMap->mvKeyFrames.push_back(mLastKeyFrame);
	mbNeedNewKF = false;
	std::cout << mpMap->mvKeyFrames.size() << std::endl;
}

#define RANSAC_MAX_ITER 35
#define RANSAC_NUM_POINTS 6
#define INLINER_THRESH 0.02
#define HIGH_PROB_DIST 3.0

bool Tracking::TrackFrame() {

	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mLastKeyFrame.mDescriptors, matches, 2);

	for(int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if(firstMatch.distance < 0.7 *  secondMatch.distance) {
				Matches.push_back(firstMatch);
		}
	}

	if(Matches.size() < 100)
		mbNeedNewKF = true;

	if(Matches.size() < 3)
		return false;

//	std::cout << "No Matches : " << Matches.size() << std::endl;
	std::vector<Eigen::Vector3d> vNextKPs, vMapKPs;
	vNextKPs.reserve(Matches.size());
	vMapKPs.reserve(Matches.size());
	std::vector<Eigen::Vector3d> pvec;
	std::vector<Eigen::Vector3d> qvec;
	Matrix3f Rp = mNextFrame.mRcw;
	float3 tp = Converter::CvMatToFloat3(mNextFrame.mtcw);
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
		MapPoint& trainPt = mLastKeyFrame.mMapPoints[trainId];

		Eigen::Vector3d p, q;
		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;
		vNextKPs.push_back(p);
		vMapKPs.push_back(q);
	}

	int best_inliners = 0;
	float best_cost = 1000;
	Eigen::Matrix3d best_R;
	Eigen::Vector3d best_t;
	for (int i = 0; i < RANSAC_MAX_ITER; ++i) {

		std::vector<int> pair;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			int index = std::rand() % vNextKPs.size();
			pair.push_back(index);
		}

		bool valid = true;
		for (int j = 0; j < RANSAC_NUM_POINTS; j++) {
			for (int k = 0; k < RANSAC_NUM_POINTS; ++k) {
				if (j == k)
					continue;
				if (pair[j] == pair[k])
					valid = false;
			}
		}
		if (!valid)
			continue;

		Eigen::Vector3d p_mean, q_mean;
		p_mean = q_mean = Eigen::Vector3d::Zero();
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {

			Eigen::Vector3d p, q;
			p = vNextKPs[pair[j]];
			q = vMapKPs[pair[j]];

			p_mean += p;
			q_mean += q;

			pvec.push_back(p);
			qvec.push_back(q);
		}

		p_mean /= RANSAC_NUM_POINTS;
		q_mean /= RANSAC_NUM_POINTS;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		float sigmap, sigmaq;
		sigmap = sigmaq = 0;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			sigmap += (pvec[j] - p_mean).norm();
			sigmaq += (qvec[j] - q_mean).norm();
			Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = U * V.transpose();
		float scale = sqrtf(sigmap / sigmaq);
		int detR = R.determinant();
		if (detR != 1)
			continue;

		Eigen::Vector3d t = p_mean - R * q_mean;

		int num_inliners = 0;
		Ab = Eigen::Matrix3d::Zero();
		p_mean = q_mean = Eigen::Vector3d::Zero();
		pvec.clear();
		qvec.clear();
		float cost = 0;
		for (int k = 0; k < vMapKPs.size(); ++k) {

			Eigen::Vector3d p, q;
			p = vNextKPs[k];
			q = vMapKPs[k];

			float dist = (p - (R * q + t)).norm();
			if (dist < INLINER_THRESH) {
				num_inliners++;
				p_mean += p;
				q_mean += q;
				pvec.push_back(p);
				qvec.push_back(q);
			}
		}

		if (num_inliners < 0.1 * Matches.size())
			continue;

		if (num_inliners >= best_inliners) {
			p_mean /= num_inliners;
			q_mean /= num_inliners;
			sigmap = sigmaq = 0;
			for (int j = 0; j < num_inliners; ++j) {
				sigmap += (pvec[j] - p_mean).norm();
				sigmaq += (qvec[j] - q_mean).norm();
				Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
			}
//			float scale = sqrtf(sigmap / sigmaq);
			best_inliners = num_inliners;
			Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d V = svd2.matrixV();
			Eigen::Matrix3d U = svd2.matrixU();
			best_R =  U * V.transpose();
			best_t = p_mean - best_R * q_mean;
//			std::cout << "scale: " << sqrtf(sigmap / sigmaq) << std::endl;
		}
	}

	if(best_inliners < 50)
		mbNeedNewKF = true;

	if(best_inliners < 0.2 * Matches.size())
		return false;

	float totalCost = 0;
	for(int i = 0; i < Matches.size(); ++i) {
		Eigen::Vector3d p, q;
		p = vNextKPs[i];
		q = vMapKPs[i];
		totalCost += (p - (best_R * q + best_t)).norm();
	}

	std::cout << "avg. Cost: " << totalCost / Matches.size() << std::endl;

//	g2o::SparseOptimizer optimizer;
//	optimizer.setVerbose(false);
//	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
//	    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
//	);
//	optimizer.setAlgorithm(solver);
//
//	Eigen::Matrix4d Tp = Converter::TransformToEigen(mLastKeyFrame.mRcw, mLastKeyFrame.mtcw);
//	Eigen::Matrix3d last_r = Tp.topLeftCorner(3, 3);
//	Eigen::Vector3d last_t = Tp.topRightCorner(3, 1);
//	Eigen::Quaterniond q(last_r);
//	g2o::SE3Quat last_pose(q, last_t);
//	g2o::VertexSE3Expmap * v_last = new g2o::VertexSE3Expmap();
//	v_last->setId(0);
//	v_last->setFixed(true);
//	v_last->setEstimate(last_pose);
//	optimizer.addVertex(v_last);

	// initialise g2o
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
	optimizer.setAlgorithm(solver);

	std::vector<Eigen::Matrix4d> poses;
	Eigen::Matrix4d Tp = Converter::TransformToEigen(mLastKeyFrame.mRcw, mLastKeyFrame.mtcw);
	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Td.topLeftCorner(3, 3) = best_R;
	Td.topRightCorner(3, 1) = best_t;
	Tc =  Td.inverse() * Tp;
	poses.push_back(Tp);
	poses.push_back(Tc);

//	for(int i = 0; i < 2; ++i) {
//		Eigen::Vector3d t = poses[i].topRightCorner(3, 1);
//		Eigen::Matrix3d r = poses[i].topLeftCorner(3, 3);
//		Eigen::Quaterniond q(r);
//		Eigen::Isometry3d cam;
//		cam = q;
//		cam.translation() = t;
//
//		g2o::VertexSE3 *vc = new g2o::VertexSE3();
//		vc->setEstimate(cam);
//		vc->setId(i);
//
//		if(i == 0)
//			vc->setFixed(true);
//
//		optimizer.addVertex(vc);
//	}

//	for(int i = 0; i < pvec.size(); ++i) {
//		g2o::VertexSE3* vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
//		g2o::VertexSE3* vp1 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);
//		Eigen::Vector3d pt0, pt1;
//		pt0 = vp0->estimate() * qvec[i];
//		pt1 = vp1->estimate() * pvec[i];
//
//		g2o::Edge_V_V_GICP * e = new g2o::Edge_V_V_GICP();
//		e->setVertex(0, vp0);
//		e->setVertex(1, vp1);
//
//	    Eigen::Vector3d nm0, nm1;
//	    nm0 << 0, i, 1;
//	    nm1 << 0, i, 1;
//	    nm0.normalize();
//	    nm1.normalize();
//
//		g2o::EdgeGICP meas;
//		meas.pos0 = pt0;
//		meas.pos1 = pt1;
//	    meas.normal0 = nm0;
//	    meas.normal1 = nm1;
//
//		e->setMeasurement(meas);
//		meas = e->measurement();
//		e->information() = meas.prec0(0.01);
//
//		optimizer.addEdge(e);
//	}

//	optimizer.initializeOptimization();
//    optimizer.computeActiveErrors();
//    std::cout << "Initial chi2 = " << std::FIXED(optimizer.chi2()) << std::endl;
//	optimizer.setVerbose(true);
//    optimizer.optimize(5);
//	Eigen::Matrix3d next_r = Tc.topLeftCorner(3, 3);
//	Eigen::Vector3d next_t = Tc.topRightCorner(3, 1);
//	Eigen::Quaterniond p(next_r);
//	g2o::SE3Quat next_pose(q, next_t);
//	g2o::VertexSE3Expmap * v_next = new g2o::VertexSE3Expmap();
//	v_next->setId(1);
//	v_next->setFixed(false);
//	v_next->setEstimate(next_pose);
//	optimizer.addVertex(v_next);
//
////	std::vector<Eigen::Vector3d> truePoints;
//	int point_id = 2;
//	for(int i = 0; i < pvec.size(); ++i) {
//	    g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
//	    v_p->setId(point_id);
//	    v_p->setMarginalized(false);
//	    v_p->setEstimate(last_r * qvec[i] + last_t);
////		Eigen::Vector3d q_g = ;
//	    optimizer.addVertex(v_p);
//
//	    float fx0 = Frame::fx(0);
//	    float fy0 = Frame::fy(0);
//	    float cx0 = Frame::cx(0);
//	    float cy0 = Frame::cy(0);
//
//	    Eigen::Vector2d zp, zq;
//	    zp<< fx0 * pvec[i](0) / pvec[i](2) + cx0, fy0 * pvec[i](1) / pvec[i](2) + cy0;
//	    zq<< fx0 * qvec[i](0) / qvec[i](2) + cx0, fy0 * qvec[i](1) / qvec[i](2) + cy0;
//
//		g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
//		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
//		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(0)->second));
//		e->setMeasurement(zq);
//		e->setParameterId(0, 0);
//
//		g2o::EdgeProjectXYZ2UV * e2 = new g2o::EdgeProjectXYZ2UV();
//		e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
//		e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(1)->second));
//		e2->setMeasurement(zp);
//		e2->setParameterId(0, 0);
//
//		optimizer.addEdge(e);
//		optimizer.addEdge(e2);
//		point_id++;
//	}
//
//	optimizer.initializeOptimization();
//	optimizer.setVerbose(true);
//	optimizer.optimize(10);

	Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
	mNextFrame.mRwc = mNextFrame.mRcw.t();

	mNoFrames++;

	return true;
}

bool Tracking::TrackMap() {

	std::vector<cv::DMatch> Matches;
	std::vector<std::vector<cv::DMatch>> matches;
	mORBMatcher->knnMatch(mNextFrame.mDescriptors, mpMap->mDescriptors, matches, 2);

	for(int i = 0; i < matches.size(); ++i) {
		cv::DMatch& firstMatch = matches[i][0];
		cv::DMatch& secondMatch = matches[i][1];
		if(firstMatch.distance < 0.6 *  secondMatch.distance) {
				Matches.push_back(firstMatch);
		}
	}

	if(Matches.size() < 3)
		return false;

	std::cout << "No Matches : " << Matches.size() << std::endl;
	std::vector<Eigen::Vector3d> vNextKPs, vMapKPs;
	vNextKPs.reserve(Matches.size());
	vMapKPs.reserve(Matches.size());
	Matrix3f Rp = mNextFrame.mRcw;
	float3 tp = Converter::CvMatToFloat3(mNextFrame.mtcw);
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
		MapPoint& trainPt = mpMap->mMapPoints[trainId];

		Eigen::Vector3d p, q;
		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;
		vNextKPs.push_back(p);
		vMapKPs.push_back(q);
	}

	int best_inliners = 0;
	float best_cost = 1000;
	Eigen::Matrix3d best_R;
	Eigen::Vector3d best_t;
	for (int i = 0; i < RANSAC_MAX_ITER; ++i) {

		std::vector<int> pair;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			int index = std::rand() % vNextKPs.size();
			pair.push_back(index);
		}

		bool valid = true;
		for (int j = 0; j < RANSAC_NUM_POINTS; j++) {
			for (int k = 0; k < RANSAC_NUM_POINTS; ++k) {
				if (j == k)
					continue;
				if (pair[j] == pair[k])
					valid = false;
			}
		}
		if (!valid)
			continue;

		std::vector<Eigen::Vector3d> pvec;
		std::vector<Eigen::Vector3d> qvec;
		Eigen::Vector3d p_mean, q_mean;
		p_mean = q_mean = Eigen::Vector3d::Zero();
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {

			Eigen::Vector3d p, q;
			p = vNextKPs[pair[j]];
			q = vMapKPs[pair[j]];

			p_mean += p;
			q_mean += q;

			pvec.push_back(p);
			qvec.push_back(q);
		}

		p_mean /= RANSAC_NUM_POINTS;
		q_mean /= RANSAC_NUM_POINTS;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		float sigmap, sigmaq;
		sigmap = sigmaq = 0;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			sigmap += (pvec[j] - p_mean).norm();
			sigmaq += (qvec[j] - q_mean).norm();
			Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = U * V.transpose();
		float scale = sqrtf(sigmap / sigmaq);
		int detR = R.determinant();
		if (detR != 1)
			continue;

		Eigen::Vector3d t = p_mean - R * q_mean;

		int num_inliners = 0;
		Ab = Eigen::Matrix3d::Zero();
		p_mean = q_mean = Eigen::Vector3d::Zero();
		pvec.clear();
		qvec.clear();
		float cost = 0;
		for (int k = 0; k < vMapKPs.size(); ++k) {

			Eigen::Vector3d p, q;
			p = vNextKPs[k];
			q = vMapKPs[k];

			float dist = (p - (R * q + t)).norm();
			if (dist < INLINER_THRESH) {
				num_inliners++;
				p_mean += p;
				q_mean += q;
				pvec.push_back(p);
				qvec.push_back(q);
			}
		}

		if (num_inliners < 0.1 * Matches.size())
			continue;

		if (num_inliners >= best_inliners) {
			p_mean /= num_inliners;
			q_mean /= num_inliners;
			sigmap = sigmaq = 0;
			for (int j = 0; j < num_inliners; ++j) {
				sigmap += (pvec[j] - p_mean).norm();
				sigmaq += (qvec[j] - q_mean).norm();
				Ab += (pvec[j] - p_mean) * (qvec[j] - q_mean).transpose();
			}
			best_inliners = num_inliners;
			Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d V = svd2.matrixV();
			Eigen::Matrix3d U = svd2.matrixU();
			best_R =  U * V.transpose();
			best_t = p_mean - best_R * q_mean;
		}
	}

	if(best_inliners < 0.1 * Matches.size())
		return false;

	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Tc.topLeftCorner(3, 3) = best_R.transpose();
	Tc.topRightCorner(3, 1) = -best_R.transpose() * best_t;

	Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
	mNextFrame.mRwc = mNextFrame.mRcw.t();

	std::vector<cv::DMatch> newMatches;
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
		MapPoint& trainPt = mpMap->mMapPoints[trainId];

		Eigen::Vector3d p, q;
		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;

		float dist = (p - (best_R * q + best_t)).norm();
		if (dist < INLINER_THRESH) {
			newMatches.push_back(Matches[i]);
		}
	}

	FuseKeyPointsAndDescriptors(mNextFrame, mpMap->mMapPoints, mpMap->mDescriptors, newMatches);

	return true;

}

void Tracking::TrackICP() {

	const float w = 0.1;
	Eigen::Matrix<double, 6, 1> result;
	Eigen::Matrix<float, 6, 6> host_a;
	Eigen::Matrix<float, 6, 1> host_b;

//	ShowResiduals();

	for(int i = 2; i >= 0; --i)
		for(int j = 0; j < iter[i]; j++) {

			cost = ICPReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
//			std::cout << "Last ICP Error: " << cost << std::endl;

			Eigen::Matrix<double, 6, 6> dA_icp = host_a.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = host_b.cast<double>();

//			cost = RGBReduceSum(mNextFrame, mLastFrame, i, host_a.data(), host_b.data());
//			std::cout << "Last RGB Error: " << cost << std::endl;

//			Eigen::Matrix<double, 6, 6> dA_rgb = host_a.cast<double>();
//			Eigen::Matrix<double, 6, 1> db_rgb = host_b.cast<double>();

//			Eigen::Matrix<double, 6, 6> dA = w * w * dA_icp + dA_rgb;
//			Eigen::Matrix<double, 6, 1> db = w * db_icp + db_rgb;
			Eigen::Matrix<double, 6, 6> dA = dA_icp;
			Eigen::Matrix<double, 6, 1> db = db_icp;
			result = dA.ldlt().solve(db);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();

			Eigen::Matrix<double, 4, 4> Tc = Converter::TransformToEigen(mNextFrame.mRcw, mNextFrame.mtcw);
			Eigen::Matrix<double, 4, 4> Tp = Converter::TransformToEigen(mLastFrame.mRcw, mLastFrame.mtcw);
//			std::cout << "T:\n" << Tc << std::endl;
			Tc = Tp * (dT.inverse() * Tc.inverse() * Tp).inverse();

			Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
			mNextFrame.mRwc = mNextFrame.mRcw.t();
	}
	ShowResiduals();
}

void Tracking::AddObservation(const Rendering& render) {
	mLastFrame = Frame(mLastFrame, render);
}

void Tracking::SetMap(Map* pMap) {
	mpMap = pMap;
}

void Tracking::ShowResiduals() {

	DeviceArray2D<uchar> warpImg(640, 480);
	DeviceArray2D<uchar> residual(640, 480);
	warpImg.zero();
	residual.zero();
	WarpGrayScaleImage(mNextFrame, mLastFrame, residual);
	ComputeResidualImage(residual, warpImg, mNextFrame);
	cv::Mat cvresidual(480, 640, CV_8UC1);
	warpImg.download((void*)cvresidual.data, cvresidual.step);
	cv::imshow("residual", cvresidual);

//	cv::Mat hist;
//	int histSize = 256;
//	float range[] = { 0, 256 } ;
//	const float* histRange = { range };
//	cv::calcHist(&cvresidual, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
//
//	int hist_w = 512;
//	int hist_h = 400;
//	int bin_w = cvRound((double) hist_w / histSize);
//	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
//			cv::Mat());
//	for (int i = 1; i < histSize; i++) {
//		cv::line(histImage,
//				cv::Point(bin_w * (i - 1),
//						hist_h - cvRound(hist.at<float>(i - 1))),
//				cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
//				cv::Scalar(255, 0, 0), 2, 8, 0);
//	}
//	cv::imshow("histImage", histImage);
//
//	int key = cv::waitKey(0);
//	if(key == 27)
//		exit(0);
//	if(key == 's') {
//		cv::imwrite("residual.jpg", cvresidual);
//	}
}

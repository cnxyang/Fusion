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
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


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

//	if(mbNeedNewKF)
		CreateKeyFrame();

	mLastFrame = Frame(mNextFrame);

	return true;
}

bool Tracking::Track() {
	bool bOK;
	switch(mNextState) {
	case NOT_INITIALISED:
		bOK = CreateInitialMap();
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

bool Tracking::CreateInitialMap() {
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
	if(!bOK)
		return false;
//	TrackICP();
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

#define RANSAC_MAX_ITER 55
#define RANSAC_NUM_POINTS 6
#define INLINER_THRESH 0.02
//#define HIGH_PROB_DIST 3.0

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

	if(Matches.size() < 6)
		return false;

	std::cout << "No Matches : " << Matches.size() << std::endl;
	std::vector<MapPoint> vNextKPs, vMapKPs;
	vNextKPs.reserve(Matches.size());
	vMapKPs.reserve(Matches.size());
	std::vector<MapPoint> pvec;
	std::vector<MapPoint> qvec;
	Matrix3f Rp = mNextFrame.mRcw;
	float3 tp = Converter::CvMatToFloat3(mNextFrame.mtcw);
	for(int i = 0; i < Matches.size(); ++i) {
		int queryId = Matches[i].queryIdx;
		int trainId = Matches[i].trainIdx;
		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
		MapPoint& trainPt = mLastKeyFrame.mvpMapPoints[trainId];

//		Eigen::Vector3d p, q;
//		p << queryPt.pos.x, queryPt.pos.y,  queryPt.pos.z;
//		q << trainPt.pos.x, trainPt.pos.y, trainPt.pos.z;
		vNextKPs.push_back(queryPt);
		vMapKPs.push_back(trainPt);
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

			MapPoint p, q;
			p = vNextKPs[pair[j]];
			q = vMapKPs[pair[j]];

			p_mean += p.pos;
			q_mean += q.pos;

			pvec.push_back(p);
			qvec.push_back(q);
		}

		p_mean /= RANSAC_NUM_POINTS;
		q_mean /= RANSAC_NUM_POINTS;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		float sigmap, sigmaq;
		sigmap = sigmaq = 0;
		for (int j = 0; j < RANSAC_NUM_POINTS; ++j) {
			sigmap += (pvec[j].pos - p_mean).norm();
			sigmaq += (qvec[j].pos - q_mean).norm();
			Ab += (pvec[j].pos - p_mean) * (qvec[j].pos - q_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = (V * U.transpose()).transpose();
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

			MapPoint p, q;
			p = vNextKPs[k];
			q = vMapKPs[k];

			float dist = (p.pos - (R * q.pos + t)).norm();
			if (dist < INLINER_THRESH) {
				num_inliners++;
				p_mean += p.pos;
				q_mean += q.pos;
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
				sigmap += (pvec[j].pos - p_mean).norm();
				sigmaq += (qvec[j].pos - q_mean).norm();
				Ab += (pvec[j].pos - p_mean) * (qvec[j].pos - q_mean).transpose();
			}
//			float scale = sqrtf(sigmap / sigmaq);
			best_inliners = num_inliners;
			Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d V = svd2.matrixV();
			Eigen::Matrix3d U = svd2.matrixU();
			best_R =  (V * U.transpose()).transpose();
			best_t = p_mean - best_R * q_mean;
//			std::cout << "scale: " << sqrtf(sigmap / sigmaq) << std::endl;
			if(best_R.determinant() < 1e-1) {
				best_R = R;
				best_t = t;
			}
		}
	}

//	if(best_inliners < 50)
//		mbNeedNewKF = true;
//
//	if(best_inliners < 50)
//		return false;
//
//	if(pvec.size() <= 0)
//		return false;

	Eigen::Matrix4d Tp = Converter::TransformToEigen(mLastKeyFrame.mRcw, mLastKeyFrame.mtcw);
	Eigen::Matrix4d Td = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
	Td.topLeftCorner(3, 3) = best_R;
	Td.topRightCorner(3, 1) = best_t;
	Tc =  Td.inverse() * Tp;

//	g2o::SparseOptimizer optimizer;
//	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
//	    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
//	);
//	optimizer.setAlgorithm(solver);
//	optimizer.setVerbose(true);
//
//	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//	g2o::SE3Quat pose(best_R, best_t);
////	g2o::SE3Quat pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
//	std::cout << "Pose Before Optimize: \n" << Td << std::endl;
//	vSE3->setEstimate(pose);
//	vSE3->setId(0);
//	vSE3->setFixed(false);
//	optimizer.addVertex(vSE3);
//
//	std::vector<g2o::EdgeSE3ProjectXYZ*> edgeList;
//	for(int i = 0; i < Matches.size(); i++) {
//		int queryId = Matches[i].queryIdx;
//		int trainId = Matches[i].trainIdx;
//		MapPoint& queryPt = mNextFrame.mMapPoints[queryId];
//		MapPoint& trainPt = mLastKeyFrame.mvpMapPoints[trainId];
//
//		g2o::VertexSBAPointXYZ * vp = new g2o::VertexSBAPointXYZ();
//		vp->setId(i+1);
//		vp->setFixed(true);
//		vp->setEstimate(trainPt.pos);
//		optimizer.addVertex(vp);
//
//        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//
//        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
//        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
//        e->setMeasurement(queryPt.uv);
//        e->setInformation(Eigen::Matrix2d::Identity());
//        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//        e->setRobustKernel(rk);
//
//        e->fx = Frame::fx(0);
//        e->fy = Frame::fy(0);
//        e->cx = Frame::cx(0);
//        e->cy = Frame::cy(0);
//
////        std::cout << "true pose: \n" << trainPt.pos << "\n"
////        		  << "observation : \n " << queryPt.uv << "\n"
////        		  << "estimation: \n " << e->cam_project(vSE3->estimate().map(vp->estimate())) << std::endl;
//
//        optimizer.addEdge(e);
//        edgeList.push_back(e);
//	}
//
//
//	for(size_t it=0; it<4; it++) {
//
//		vSE3->setEstimate(pose);
//		optimizer.initializeOptimization(0);
//		optimizer.optimize(10);
//		for(size_t i=0, iend=edgeList.size(); i<iend; i++) {
//			g2o::EdgeSE3ProjectXYZ* e = edgeList[i];
//			if(e->level() == 0) {
//				e->computeError();
//			}
//			const float chi2 = e->chi2();
//			if(chi2 > 0.89) {
//				e->setLevel(1);
//			}
//		}
//	}
//
//	int inliners = 0;
//	for(size_t i=0, iend=edgeList.size(); i<iend; i++) {
//		g2o::EdgeSE3ProjectXYZ* e = edgeList[i];
//		if(e->level() == 0) {
//			inliners++;
//		}
//	}
//
//	std::cout << "inliners: " << inliners << std::endl;
//
//    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
//    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
//    Eigen::Matrix<double,4,4> eigMat = SE3quat_recov.to_homogeneous_matrix();
//    Tc = eigMat.inverse() * Tp;
    Converter::TransformToCv(Tc, mNextFrame.mRcw, mNextFrame.mtcw);
    mNextFrame.mRwc = mNextFrame.mRcw.t();

	std::cout << std::endl << "After Optimization: \n" << Tc << std::endl;
	mNoFrames++;

	return true;
}

bool Tracking::TrackMap() {

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

#include "Frame.h"
#include "AOTracker.h"
#include "Settings.h"
#include <chrono>
#include <random>

AOTracker::AOTracker(int w, int h, Eigen::Matrix3f K) :
	width(w), height(h), trackingWasGood(false)
{
	SURF = cv::cuda::SURF_CUDA(20);
	matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
}

SE3 AOTracker::trackFrame(Frame* frame, Frame* ref, int iterations)
{
	if(!frame->keyPointStruct)
		extractKeyPoints(frame);

	if(!ref->keyPointStruct)
		extractKeyPoints(ref);

	std::vector<cv::DMatch> matchResults;
	std::vector<std::vector<cv::DMatch>> matches;
	std::vector<Eigen::Vector3d> queryPoints;
	std::vector<Eigen::Vector3d> trainPoints;

	matcher->knnMatch(frame->keyPointStruct->descriptors, ref->keyPointStruct->descriptors, matches, 2);
	// filter out useful key point matches
	for (auto match : matches)
	{
		cv::DMatch& first = match[0];
		cv::DMatch& second = match[1];
		if(first.distance < 0.9 * second.distance)
			matchResults.push_back(first);
	}

	// Get rid of invalid points;
	int NMatches = 0;
	for (auto m : matchResults)
	{
		if(frame->keyPointStruct->valid[m.queryIdx]
		   && ref->keyPointStruct->valid[m.trainIdx])
		{
			queryPoints.push_back(frame->keyPointStruct->pt3d[m.queryIdx]);
			trainPoints.push_back(ref->keyPointStruct->pt3d[m.trainIdx]);
			NMatches++;
		}
	}

	if(NMatches < 50) return SE3();

	// Initialise the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(1, NMatches - 1);

	// Generate random samples
	std::vector<Eigen::Vector3d> querySample;
	std::vector<Eigen::Vector3d> trainSample;
	for (int i = 0; i < iterations; ++i)
	{
		std::vector<int> indices;
		for (int j = 0; j < 3; ++j)
		{
			// Generate random indices
			indices.push_back(dis(gen));
			// Make sure all 3 points are different
			for (int k = 0; k < j; ++k)
				if(indices[k] == indices[j])
					indices[j] = (indices[j] + 1) % NMatches;
		}

		for (auto index : indices)
		{
			querySample.push_back(queryPoints[index]);
			trainSample.push_back(trainPoints[index]);
		}
	}

	std::vector<Eigen::Matrix4d> PoseHypotheses;
	for (int i = 0; i < querySample.size(); i += 3)
	{
		Eigen::Vector3d queryMean = Eigen::Vector3d::Zero();
		Eigen::Vector3d trainMean = Eigen::Vector3d::Zero();
		for(int j = 0; j < 3; ++j)
		{
			queryMean += querySample[i + j];
			trainMean += trainSample[i + j];
		}

		queryMean /= 3;
		trainMean /= 3;

		for(int j = 0; j < 3; ++j)
		{
			querySample[i + j] -= queryMean;
			trainSample[i + j] -= trainMean;
		}

		// make sure they are not co-linear
		float queryDist = (querySample[i + 1] - querySample[i]).cross(querySample[i] - querySample[i + 2]).norm();
		float trainDist = (trainSample[i + 1] - trainSample[i]).cross(trainSample[i] - trainSample[i + 2]).norm();
		if (queryDist < 1e-3 || trainDist < 1e-3) continue;

		Eigen::Matrix3d matAb = Eigen::Matrix3d::Zero();
		for (int j = 0; j < 3; ++j)
			matAb += querySample[i + j] * trainSample[i + j].transpose();

		// SVD matrix factorisation
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(matAb, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();

		// Make sure we have a valid rotation matrix
		Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
		if ((V * U.transpose()).determinant() < 0) I(2, 2) = -1;
		Eigen::Matrix3d R = (V * I * U.transpose()).transpose();
		Eigen::Vector3d t = queryMean - R * trainMean;
		Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
		T.topLeftCorner(3, 3) = R;
		T.topRightCorner(3, 1) = t;
		PoseHypotheses.push_back(T);
	}

	SE3 BestPose = SE3();
	int BestNInlier = 0;
	for (auto pose : PoseHypotheses)
	{
		Eigen::Matrix3d R = pose.topLeftCorner(3, 3);
		Eigen::Vector3d t = pose.topRightCorner(3, 1);
		int NInlier = 0;
		for(int i = 0; i < querySample.size(); ++i)
		{
			double dist = (querySample[i] - (R * trainSample[i] + t)).norm();
			if(dist <= 0.1)
				NInlier++;
		}

		if(NInlier > BestNInlier)
		{
			BestPose = SE3(R, t);
		}
	}

	ref->keyPointStruct->minimizeMemoryFootprint();
	return BestPose;
}

void AOTracker::extractKeyPoints(Frame* frame)
{
	frame->keyPointStruct = new KeyPointStruct();

	// Convert input images to gray-scale images
	cv::Mat image;
	cv::cvtColor(frame->data.image, image, cv::COLOR_RGB2GRAY);

	// Detect and compute feature descriptors
	cv::cuda::GpuMat img(image);
	SURF(img, cv::cuda::GpuMat(), frame->keyPointStruct->keyPoints, frame->keyPointStruct->descriptors);

	// Extract depth from depth input
	float depthScale = systemState.depthScale;
	for (auto keyPoint : frame->keyPointStruct->keyPoints)
	{
		int ix = (int)(keyPoint.pt.x + 0.5);
		int iy = (int)(keyPoint.pt.y + 0.5);
		float z = (float)frame->data.depth.at<unsigned short>(iy, ix) / depthScale;
		frame->keyPointStruct->depth.push_back(z);

		if(z > 0)
			frame->keyPointStruct->valid.push_back(true);
		else
			frame->keyPointStruct->valid.push_back(false);

		float x = z * (keyPoint.pt.x - frame->cx()) * frame->fxInv();
		float y = z * (keyPoint.pt.y - frame->cy()) * frame->fyInv();
		Eigen::Vector3d pt;
		pt << x, y, z;
		frame->keyPointStruct->pt3d.push_back(pt);
	}
}

#include "KeyFrameGraph.h"

KeyFrameGraph::KeyFrameGraph(int w, int h, Eigen::Matrix3f K)
{
	fowX = 2 * atanf((float)((w / K(0,0)) / 2.0f));
	fowY = 2 * atanf((float)((h / K(1,1)) / 2.0f));
}

KeyFrameGraph::~KeyFrameGraph()
{

}

void KeyFrameGraph::addKeyFrame(Frame* frame)
{
	keyFramesAll.push_back(frame);
}

std::unordered_set<Frame*, std::hash<Frame*>> KeyFrameGraph::searchCandidates(Frame* kf)
{
	std::unordered_set<Frame*, std::hash<Frame*>> result;

//	std::vector<TrackableKFStruct> potentialReferenceFrames = findEuclideanOverlapFrames(kf);

	return result;
}

std::vector<TrackableKFStruct> KeyFrameGraph::findEuclideanOverlapFrames(Frame* kf, float distTH, float angleTH)
{
	float cosAngleTH = cosf(angleTH * 0.5f * (fowX + fowY));

	Eigen::Vector3d pos = kf->getCamToWorld().translation();
	Eigen::Vector3d viewingDir = kf->getCamToWorld().rotationMatrix().rightCols<1>();

	std::vector<TrackableKFStruct> result;
	keyFramesAllMutex.lock();
	for(unsigned int i = 0; i < keyFramesAll.size(); ++i)
	{
		Eigen::Vector3d otherPos = keyFramesAll[i]->getCamToWorld().translation();
		Eigen::Vector3d dist = (pos - otherPos);
		float dNorm2 = dist.dot(dist);
		if(dNorm2 > distTH) continue;

		Eigen::Vector3d otherViewingDir = keyFramesAll[i]->getCamToWorld().rotationMatrix().rightCols<1>();
		float dirDotProd = otherViewingDir.dot(viewingDir);
		if(dirDotProd < cosAngleTH) continue;

		result.push_back(TrackableKFStruct());
	}

	return result;
}

std::vector<SE3> KeyFrameGraph::getAllKeyFramePoses() const
{
	std::vector<SE3> poses;
	std::transform(keyFramesAll.begin(), keyFramesAll.end(), std::back_inserter(poses), [](Frame* f) { return f->pose; });
	return poses;
}

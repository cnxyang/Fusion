#include "keyFrame.h"

KeyFrame::KeyFrame(const Frame * src) :
		valid(true), N(src->N),frameId(src->frameId) {

	pose = src->pose;
	frameKeys = src->mPoints;
	frameDescriptors = src->descriptors;
}

Eigen::Matrix3d KeyFrame::rotation() const {
	return pose.topLeftCorner(3, 3);
}

Eigen::Vector3d KeyFrame::translation() const {
	return pose.topRightCorner(3, 1);
}

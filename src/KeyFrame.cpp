#include "KeyFrame.hpp"

KeyFrame::KeyFrame(const Frame * src) :
		valid(true), N(src->mNkp),frameId(src->frameId) {

	pose = src->mPose;
	frameKeys = src->mPoints;
	frameDescriptors = src->mDescriptors;
}

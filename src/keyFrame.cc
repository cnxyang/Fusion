#include "KeyFrame.hpp"

KeyFrame::KeyFrame(const Frame * src) :
		valid(true), N(src->N),frameId(src->frameId) {

	pose = src->pose;
	frameKeys = src->mPoints;
	frameDescriptors = src->descriptors;
}

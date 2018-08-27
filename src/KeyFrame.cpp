#include "KeyFrame.hpp"

int KeyFrame::nextId = 0;

KeyFrame::KeyFrame(const Frame& F) {
	mKFId = nextId++;
}

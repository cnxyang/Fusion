#include "KeyFrame.h"

KeyFrame::KeyFrame(const Frame& frame) {
	mRcw = frame.mRcw.clone();
	mtcw = frame.mtcw.clone();
}

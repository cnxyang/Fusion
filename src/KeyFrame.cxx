#include "KeyFrame.h"

KeyFrame::KeyFrame() {

}

KeyFrame::KeyFrame(const Frame& frame) {
	mRcw = frame.mRcw.clone();
	mtcw = frame.mtcw.clone();
	mRwc = frame.mRwc.clone();
	frame.mDescriptors.copyTo(mDescriptors);
	mvpMapPoints = frame.mMapPoints;
}

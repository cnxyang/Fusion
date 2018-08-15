#include "Viewer.hpp"
#include <algorithm>
#include <pangolin/pangolin.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer():
mpMap(nullptr),
mpTracker(nullptr),
mpSystem(nullptr) {

}

void Viewer::Spin() {

//	CreateWindowAndBind("main", 1920, 1080);
//	glEnable(GL_DEPTH_TEST);
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//	OpenGlRenderState s_cam(
//			ProjectionMatrix(mCols, mRows, 525, 525, mCols / 2, mRows / 2, 0.1, 1000),
//			ModelViewLookAt(0, 1, -3, 0, 1, 0, AxisX));
//
//	View& d_cam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(300), 1.0, -mAspectRatio).SetHandler(new Handler3D(s_cam));
//	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(300));
//	Var<bool> a_button("UI.A_Button", false, false);
//
//	View& d_img = Display("image").SetBounds(1.0, Attach::Pix(-360), 1.0, Attach::Pix(-480))
//		.SetLock(LockRight, LockTop);
//
//	View& d_depth = Display("depth").SetBounds(Attach::Pix(-360), 1.0, 1.0, Attach::Pix(-480))
//		.SetLock(LockRight, LockBottom);
//
//
//	unsigned char* imageArray = new unsigned char[3 * mCols * mRows];
//	unsigned short* depthArray = new unsigned short[mCols * mRows];
//	GlTexture imageTexture(mCols,mRows,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
//
//	while (1) {
//
//		if(ShouldQuit()) {
//			return;
//		}
//
//		if(pangolin::Pushed(a_button))
//			cout << "you pushed a button" << endl;
//
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//		d_cam.Activate(s_cam);
//		DrawCameraMovement();
//		DrawKeyPoints();
//
//		SetImageData(imageArray, depthArray);
//		imageTexture.Upload(imageArray, GL_RGB, GL_UNSIGNED_BYTE);
//
//		d_img.Activate();
//	    glColor3f(1.0,1.0,1.0);
//	    imageTexture.RenderToViewport();
//
//		FinishFrame();
//	}
//
//	delete imageArray;
//	delete depthArray;
}

void Viewer::SetMap(Mapping* pMap) {
	mpMap = pMap;
}

void Viewer::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}

void Viewer::SetTracker(Tracking* pTracker) {
	mpTracker = pTracker;
}

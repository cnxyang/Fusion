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

	CreateWindowAndBind("main", 1280, 960);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	OpenGlRenderState s_cam(
			ProjectionMatrix(640, 480, 525, 525, 320, 240, 0.1f, 100.0f),
			ModelViewLookAt(0, 1, -3, 0, 1, 0, AxisX)
	);

	View& d_cam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(200), 1.0, -640.0 / 480).SetHandler(new Handler3D(s_cam));
	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200));
	Var<bool> a_button("UI.A_Button", false, false);
//
//	View& d_img = Display("image").SetBounds(1.0, Attach::Pix(-360), 1.0, Attach::Pix(-480))
//		.SetLock(LockRight, LockTop);
//
//	View& d_depth = Display("depth").SetBounds(Attach::Pix(-360), 1.0, 1.0, Attach::Pix(-480))
//		.SetLock(LockRight, LockBottom);

	while (1) {

		if(ShouldQuit()) {
			return;
		}

		if(pangolin::Pushed(a_button))
			cout << "you pushed a button" << endl;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);

		FinishFrame();
	}
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

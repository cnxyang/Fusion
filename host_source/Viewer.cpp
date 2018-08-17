#include "Viewer.hpp"
#include <algorithm>

using namespace std;
using namespace pangolin;

Viewer::Viewer() :
		mpMap(nullptr), mpTracker(nullptr), mpSystem(nullptr) {
}

void Viewer::Spin() {

	CreateWindowAndBind("main", 1280, 960);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	OpenGlRenderState s_cam(
			ProjectionMatrix(640, 480, 525, 525, 320, 240, 0.1f, 1000.0f),
			ModelViewLookAt(0, 1, -1, 0, 1, 0, AxisX));

	View& d_cam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(200), 1.0,
			-640.0 / 480).SetHandler(new Handler3D(s_cam));

	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200));
	Var<bool> a_button("UI.A_Button", false, false);

	while (1) {

		if (ShouldQuit()) {
			return;
		}

		if (pangolin::Pushed(a_button))
			cout << "you pushed a button" << endl;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);

		DrawKeys();
		DrawCamera();

		FinishFrame();
	}
}

void Viewer::Insert(std::vector<GLfloat>& vPt, Eigen::Vector3d& pt) {

	vPt.push_back(pt(0));
	vPt.push_back(pt(1));
	vPt.push_back(pt(2));
}

void Viewer::DrawCamera() {

	vector<GLfloat> cam;
	Eigen::Vector3d p[5];
	p[0] << 0.05, 0.05, 0;
	p[1] << 0.05, -0.05, 0;
	p[2] << -0.05, 0.05, 0;
	p[3] << -0.05, -0.05, 0;
	p[4]<< 0, 0, -0.04;

	Eigen::Matrix3d R = mpTracker->mLastFrame.Rotation();
	Eigen::Vector3d t = mpTracker->mLastFrame.Translation();

	for(int i = 0; i < 5; ++i) {
		p[i] = R * p[i] + t;
	}

	Insert(cam, p[0]);
	Insert(cam, p[1]);
	Insert(cam, p[2]);
	Insert(cam, p[1]);
	Insert(cam, p[2]);
	Insert(cam, p[3]);
	Insert(cam, p[0]);
	Insert(cam, p[2]);
	Insert(cam, p[3]);
	Insert(cam, p[0]);
	Insert(cam, p[1]);
	Insert(cam, p[4]);
	Insert(cam, p[0]);
	Insert(cam, p[2]);
	Insert(cam, p[4]);
	Insert(cam, p[1]);
	Insert(cam, p[3]);
	Insert(cam, p[4]);
	Insert(cam, p[2]);
	Insert(cam, p[3]);
	Insert(cam, p[4]);

	glColor3f(1.0, 0.0, 1.0);
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glDrawVertices(cam.size()/3, (GLfloat*)&cam[0], GL_TRIANGLES, 3);
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}

void Viewer::DrawKeys() {

	vector<GLfloat> vGLKeys;
	vector<ORBKey> vKeys;
	mpMap->GetKeysHost(vKeys);
	for (int i = 0; i < vKeys.size(); ++i) {
		ORBKey& key = vKeys[i];
		vGLKeys.push_back(key.pos.x);
		vGLKeys.push_back(key.pos.y);
		vGLKeys.push_back(key.pos.z);
	}

	glColor3f(1.0, 1.0, 1.0);
	glDrawVertices(vGLKeys.size() / 3, (GLfloat*) &vGLKeys[0], GL_POINTS, 3);
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

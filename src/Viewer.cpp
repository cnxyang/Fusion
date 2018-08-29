#include "Viewer.hpp"
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer() :
		mpMap(nullptr), mpTracker(nullptr), mpSystem(nullptr) {
}

void Viewer::Spin() {

	CreateWindowAndBind("main", 1920, 1200);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Eigen::Matrix4f viewMatrix;
	viewMatrix << 1, 0, 0, 0,
			      0,-1, 0, 0,
			      0, 0,-1,-2,
			      0, 0, 0, 1;

	OpenGlMatrix openglViewMatrix(viewMatrix);
	GlBufferCudaPtr vertex_array(GlArrayBuffer, DeviceMap::MaxTriangles,
			GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);

	OpenGlRenderState s_cam(
			ProjectionMatrix(640, 480, 525, 525, 320, 240, 0.1f, 1000.0f),
			openglViewMatrix);

	View& d_cam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(200), 1.0,
			-640.0 / 480).SetHandler(new Handler3D(s_cam));

	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200));
	Var<bool> reset_btn("UI.Reset System", false, false);
	Var<bool> traj_btn("UI.Show Trajectory",false,true);
	Var<bool> kp_btn("UI.Show Key Points",true,true);
	Var<bool> mesh_btn("UI.Show Mesh", false, true);

	while (1) {

		if (ShouldQuit()) {
			mpSystem->Stop();
			return;
		}

		if (Pushed(reset_btn)) {
			cout << "requesting system reboot.." << endl;
			mpSystem->Reboot();
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);

		if (traj_btn)
			DrawTrajectory();

		if (kp_btn)
			DrawKeys();

		if(mesh_btn)
			DrawMesh();

		DrawCamera();

		FinishFrame();
	}
}

void Viewer::DrawMesh() {
	glColor3f(0.5, 1.0, 1.0);
	glDrawVertices(mpMap->nTriangle * 3, (GLfloat*)mpMap->mHostMesh, GL_TRIANGLES, 3);
}

void Viewer::Insert(std::vector<GLfloat>& vPt, Eigen::Vector3d& pt) {

	vPt.push_back(pt(0));
	vPt.push_back(pt(1));
	vPt.push_back(pt(2));
}

void Viewer::DrawTrajectory() {

	vector<GLfloat> vPos;
	vector<Eigen::Vector3d> trace = mpMap->GetCamTrace();
	for(int i = 0; i < trace.size(); ++i) {
		Insert(vPos, trace[i]);
	}

	glColor3f(0.0, 0.0, 1.0);
	glDrawVertices(vPos.size()/3, (GLfloat*)&vPos[0], GL_LINE_STRIP, 3);
}

void Viewer::DrawCamera() {


	vector<GLfloat> cam;
	Eigen::Vector3d p[5];
	p[0] << 0.1, 0.08, 0;
	p[1] << 0.1, -0.08, 0;
	p[2] << -0.1, 0.08, 0;
	p[3] << -0.1, -0.08, 0;
	p[4] << 0, 0, -0.08;

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

	bool lost = (mpTracker->mNextState == mpTracker->LOST);
	if(lost)
		glColor3f(1.0, 0.0, 0.0);
	else
		glColor3f(0.0, 1.0, 0.0);
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

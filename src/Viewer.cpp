#include "Viewer.hpp"
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer()
:mpMap(nullptr), mpTracker(nullptr), mpSystem(nullptr),
 mbShowMesh(false){
}

void Viewer::Spin() {

	CreateWindowAndBind("main", 1920, 1200);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	mPhongProg.AddShaderFromFile(GlSlVertexShader, "shader/VertexShader.glsl");
	mPhongProg.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	mPhongProg.Link();

	mNormalProg.AddShaderFromFile(GlSlVertexShader, "shader/NormalShader_vertex.glsl");
	mNormalProg.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	mNormalProg.Link();

	mColorProg.AddShaderFromFile(GlSlVertexShader, "shader/ColorShader_vertex.glsl");
	mColorProg.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	mColorProg.Link();

	s_cam = OpenGlRenderState(
			ProjectionMatrix(640, 480, 525, 525, 320, 240, 0.1f, 1000.0f),
			ModelViewLookAtRUB(0, 0, 0, 0, 0, 1, 0, -1, 0));

	glGenVertexArrays(1, &vao);
	array.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 6,
			GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	var = new CudaScopedMappedPtr(array);
	normal.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 3,
			GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	nvar = new CudaScopedMappedPtr(normal);
	color.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 3,
			GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	cvar = new CudaScopedMappedPtr(color);

	View& d_cam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(200), 1.0,
			-640.0 / 480).SetHandler(new Handler3D(s_cam));
	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200));
	Var<bool> BtnReset("UI.Reset System", false, false);
	Var<bool> BtnShowTrajectory("UI.Show Trajectory",false,true);
	Var<bool> BtnShowKeyPoint("UI.Show Key Points",true,true);
	Var<bool> BtnShowMesh("UI.Show Mesh", false, true);
	Var<bool> BtnShowCam("UI.Show Camera", true, true);
	Var<bool> BtnFollowCam("UI.Fllow Camera", false, true);
	Var<bool> BtnShowNormal("UI.Show Normal", false, true);
	Var<bool> BtnShowColor("UI.Show Color Map", false, true);

	while (1) {

		T = s_cam.GetModelViewMatrix().Inverse();

		if (ShouldQuit()) {
			mpSystem->Stop();
			exit(0);
		}

		if (Pushed(BtnReset)) {
			cout << "requesting system reboot.." << endl;
			mpSystem->Reboot();
		}

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);

		if (BtnShowTrajectory)
			DrawTrajectory();

		if (BtnShowKeyPoint)
			DrawKeys();

		if (BtnShowMesh) {
			if(BtnShowNormal)
				BtnShowNormal = false;
			DrawMesh(false);
		}

		if (BtnShowNormal) {
			if(BtnShowMesh)
				BtnShowMesh = false;
			DrawMesh(true);
		}

		if (BtnShowColor) {
			DrawColor();
		}

		if (BtnShowCam)
			DrawCamera();

		if (BtnFollowCam)
			FollowCam();

		FinishFrame();
	}
}

void Viewer::DrawColor() {
	if(mpMap->bUpdated) {
		cudaMemcpy((void*) **var, (void*) mpMap->mMesh,
				sizeof(float3) * mpMap->nTriangle * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **cvar, (void*) mpMap->mColorMap,
				sizeof(uchar3) * mpMap->nTriangle * 3, cudaMemcpyDeviceToDevice);
		mpMap->mMutexMesh.lock();
		mpMap->bUpdated = false;
		mpMap->mMutexMesh.unlock();
	}

	mColorProg.SaveBind();
	mColorProg.SetUniform("viewMat", s_cam.GetModelViewMatrix());
	mColorProg.SetUniform("projMat", s_cam.GetProjectionMatrix());
	glBindVertexArray(vao);
	array.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	array.Unbind();

	color.Bind();
	glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	color.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, mpMap->nTriangle * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	mColorProg.Unbind();
	glBindVertexArray(0);
}

void Viewer::FollowCam() {
	Eigen::Matrix3d R = mpTracker->mNextFrame.Rotation();
	Eigen::Vector3d t = mpTracker->mNextFrame.Translation();
	Eigen::Vector3d up = { 0, -1, 0 };
	Eigen::Vector3d eye = { 0, 0, 0 };
	Eigen::Vector3d look = { 0, 0, 1 };
	up = R * up + t;
	eye = R * eye + t;
	look = R * look + t;
	s_cam.SetModelViewMatrix(ModelViewLookAtRUB(eye(0), eye(1), eye(2), look(0), look(1),
					look(2), up(0), up(1), up(2)));
}

void Viewer::DrawMesh(bool bNormal) {

	if(mpMap->nTriangle == 0)
		return;

	if(mpMap->bUpdated) {
		cudaMemcpy((void*) **var, (void*) mpMap->mMesh,
				sizeof(float3) * mpMap->nTriangle * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **nvar, (void*) mpMap->mMeshNormal,
				sizeof(float3) * mpMap->nTriangle * 3, cudaMemcpyDeviceToDevice);
		mpMap->mMutexMesh.lock();
		mpMap->bUpdated = false;
		mpMap->mMutexMesh.unlock();
	}

	pangolin::GlSlProgram* program;
	if(bNormal)
		program = &mNormalProg;
	else
		program = &mPhongProg;

	program->SaveBind();
	program->SetUniform("viewMat", s_cam.GetModelViewMatrix());
	program->SetUniform("projMat", s_cam.GetProjectionMatrix());
	glBindVertexArray(vao);
	array.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	array.Unbind();

	normal.Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, 0);
	glEnableVertexAttribArray(1);
	normal.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, mpMap->nTriangle * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	program->Unbind();
	glBindVertexArray(0);
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

	glColor3f(1.0, 0.0, 0.0);
	glPointSize(3.0);
	glDrawVertices(vGLKeys.size() / 3, (GLfloat*) &vGLKeys[0], GL_POINTS, 3);
	glPointSize(1.0);
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

#include "Viewer.hpp"
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer() :
		mpMap(nullptr), ptracker(nullptr), psystem(nullptr),
		vertexMaped(nullptr), normalMaped(nullptr), colorMaped(nullptr) {
}

void Viewer::spin() {

	CreateWindowAndBind("main", 1920, 1200);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	phongShader.AddShaderFromFile(GlSlVertexShader, "shader/VertexShader.glsl");
	phongShader.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	phongShader.Link();

	normalShader.AddShaderFromFile(GlSlVertexShader, "shader/NormalShader_vertex.glsl");
	normalShader.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	normalShader.Link();

	colorShader.AddShaderFromFile(GlSlVertexShader, "shader/ColorShader_vertex.glsl");
	colorShader.AddShaderFromFile(GlSlFragmentShader, "shader/FragmentShader.glsl");
	colorShader.Link();

	sCam = OpenGlRenderState(
			ProjectionMatrix(640, 480, 525, 525, 320, 240, 0.1f, 1000.0f),
			ModelViewLookAtRUB(0, 0, 0, 0, 0, 1, 0, -1, 0));

	glGenVertexArrays(1, &vao);
	vertex.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 6,
			GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	vertexMaped = new CudaScopedMappedPtr(vertex);

	normal.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 3,
			GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	normalMaped = new CudaScopedMappedPtr(normal);

	color.Reinitialise(GlArrayBuffer, DeviceMap::MaxTriangles * 3,
			GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	colorMaped = new CudaScopedMappedPtr(color);

	View& dCam = CreateDisplay().
			SetBounds(0.0, 1.0, Attach::Pix(200), 1.0, -640.0 / 480).
			SetHandler(new Handler3D(sCam));

	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200));
	Var<bool> btnReset("UI.Reset System", false, false);
	Var<bool> btnShowTrajectory("UI.Show Trajectory",false,true);
	Var<bool> btnShowKeyPoint("UI.Show Key Points",true,true);
	Var<bool> btnShowMesh("UI.Show Mesh", false, true);
	Var<bool> btnShowCam("UI.Show Camera", true, true);
	Var<bool> btnFollowCam("UI.Fllow Camera", false, true);
	Var<bool> btnShowNormal("UI.Show Normal", false, true);
	Var<bool> btnShowColor("UI.Show Color Map", false, true);
	Var<bool> btnSaveMesh("UI.Save as Mesh", false, false);

	while (true) {

		if (ShouldQuit()) {
			psystem->Stop();
			exit(0);
		}

		glClearColor(0.0f, 0.2f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		dCam.Activate(sCam);

		if (Pushed(btnReset)) {
			psystem->Reboot();
		}

		if (Pushed(btnSaveMesh)) {
			psystem->mutexReq.lock();
			psystem->requestSaveMesh = true;
			psystem->mutexReq.unlock();
		}

		if (btnShowTrajectory)
			drawTrajectory();

		if (btnShowKeyPoint)
			drawKeys();

		if (btnShowMesh) {
			if (btnShowNormal)
				btnShowNormal = false;
			if (btnShowColor)
				btnShowColor = false;
			drawMesh(false);
		}

		if (btnShowNormal) {
			if (btnShowColor)
				btnShowColor = false;
			drawMesh(true);
		}

		if (btnShowColor) {
			drawColor();
		}

		if (btnShowCam)
			drawCamera();

		if (btnFollowCam)
			followCam();

		FinishFrame();
	}
}

void Viewer::drawColor() {
	if(mpMap->meshUpdated) {
		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex,
				sizeof(float3) * mpMap->noTriangles[0] * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **colorMaped, (void*) mpMap->modelColor,
				sizeof(uchar3) * mpMap->noTriangles[0] * 3, cudaMemcpyDeviceToDevice);
		mpMap->mutexMesh.lock();
		mpMap->meshUpdated = false;
		mpMap->mutexMesh.unlock();
	}

	colorShader.SaveBind();
	colorShader.SetUniform("viewMat", sCam.GetModelViewMatrix());
	colorShader.SetUniform("projMat", sCam.GetProjectionMatrix());
	glBindVertexArray(vao);
	vertex.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	vertex.Unbind();

	color.Bind();
	glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	color.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, mpMap->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	colorShader.Unbind();
	glBindVertexArray(0);
}

void Viewer::followCam() {

	Eigen::Matrix4f pose = ptracker->getCurrentPose();
	Eigen::Matrix3f rotation = pose.topLeftCorner(3, 3);
	Eigen::Vector3f translation = pose.topRightCorner(3, 1);
	Eigen::Vector3f up = { 0, -1, 0 };
	Eigen::Vector3f eye = { 0, 0, 0 };
	Eigen::Vector3f look = { 0, 0, 1 };
	up = rotation * up + translation;
	eye = rotation * eye + translation;
	look = rotation * look + translation;
	sCam.SetModelViewMatrix(ModelViewLookAtRUB(eye(0), eye(1), eye(2),
			look(0), look(1), look(2), up(0), up(1), up(2)));
}

void Viewer::drawMesh(bool bNormal) {

	if(mpMap->noTrianglesHost == 0)
		return;

	if(mpMap->meshUpdated) {
		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex,
				sizeof(float3) * mpMap->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **normalMaped, (void*) mpMap->modelNormal,
				sizeof(float3) * mpMap->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		mpMap->mutexMesh.lock();
		mpMap->meshUpdated = false;
		mpMap->mutexMesh.unlock();
	}

	pangolin::GlSlProgram* program;
	if(bNormal)
		program = &normalShader;
	else
		program = &phongShader;

	program->SaveBind();
	program->SetUniform("viewMat", sCam.GetModelViewMatrix());
	program->SetUniform("projMat", sCam.GetProjectionMatrix());
	glBindVertexArray(vao);
	vertex.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	vertex.Unbind();

	normal.Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, 0);
	glEnableVertexAttribArray(1);
	normal.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, mpMap->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	program->Unbind();
	glBindVertexArray(0);
}

void Viewer::Insert(std::vector<GLfloat>& vPt, Eigen::Vector3f& pt) {

	vPt.push_back(pt(0));
	vPt.push_back(pt(1));
	vPt.push_back(pt(2));
}

void Viewer::drawTrajectory() {

//	vector<GLfloat> vPos;
//	vector<Eigen::Vector3d> trace = mpMap->GetCamTrace();
//	for (int i = 0; i < trace.size(); ++i) {
//		Insert(vPos, trace[i]);
//	}
//
//	glColor3f(0.0, 0.0, 1.0);
//	glDrawVertices(vPos.size() / 3, (GLfloat*) &vPos[0], GL_LINE_STRIP, 3);
}

void Viewer::drawCamera() {


	vector<GLfloat> cam;
	Eigen::Vector3f p[5];
	p[0] << 0.1, 0.08, 0;
	p[1] << 0.1, -0.08, 0;
	p[2] << -0.1, 0.08, 0;
	p[3] << -0.1, -0.08, 0;
	p[4] << 0, 0, -0.08;

	Eigen::Matrix4f pose = ptracker->getCurrentPose();
	Eigen::Matrix3f rotation = pose.topLeftCorner(3, 3);
	Eigen::Vector3f translation = pose.topRightCorner(3, 1);
	for(int i = 0; i < 5; ++i) {
		p[i] = rotation * p[i] + translation;
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
	bool lost = (ptracker->state == -1);
	if(lost)
		glColor3f(1.0, 0.0, 0.0);
	else
		glColor3f(0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glDrawVertices(cam.size()/3, (GLfloat*)&cam[0], GL_TRIANGLES, 3);
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}

void Viewer::drawKeys() {

//	vector<GLfloat> vGLKeys;
//	vector<ORBKey> vKeys;
//	mpMap->GetKeysHost(vKeys);
//	for (int i = 0; i < vKeys.size(); ++i) {
//		ORBKey& key = vKeys[i];
//		vGLKeys.push_back(key.pos.x);
//		vGLKeys.push_back(key.pos.y);
//		vGLKeys.push_back(key.pos.z);
//	}
//
//	glColor3f(1.0, 0.0, 0.0);
//	glPointSize(3.0);
//	glDrawVertices(vGLKeys.size() / 3, (GLfloat*) &vGLKeys[0], GL_POINTS, 3);
//	glPointSize(1.0);
}

void Viewer::setMap(Mapping* pMap) {
	mpMap = pMap;
}

void Viewer::setSystem(System* pSystem) {
	psystem = pSystem;
}

void Viewer::setTracker(tracker* pTracker) {
	ptracker = pTracker;
}

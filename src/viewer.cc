#include "viewer.h"
#include "keyFrame.h"
#include <unistd.h>
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer() :
		mpMap(nullptr), ptracker(nullptr), psystem(nullptr), vao(0), vertexMaped(
				nullptr), normalMaped(nullptr), colorMaped(nullptr), quit(false) {
}

void Viewer::signalQuit() {
	quit = true;
}

void setImageData(unsigned char * imageArray, int size) {
	for (int i = 0; i < size; i++) {
		imageArray[i] = (unsigned char) (rand() / (RAND_MAX / 255.0));
	}
}

void Viewer::spin() {

	CreateWindowAndBind("main", 1920, 1200);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	phongShader.AddShaderFromFile(GlSlVertexShader, "shader/VertexShader.glsl");
	phongShader.AddShaderFromFile(GlSlFragmentShader,
			"shader/FragmentShader.glsl");
	phongShader.Link();

	normalShader.AddShaderFromFile(GlSlVertexShader,
			"shader/NormalShader_vertex.glsl");
	normalShader.AddShaderFromFile(GlSlFragmentShader,
			"shader/FragmentShader.glsl");
	normalShader.Link();

	colorShader.AddShaderFromFile(GlSlVertexShader,
			"shader/ColorShader_vertex.glsl");
	colorShader.AddShaderFromFile(GlSlFragmentShader,
			"shader/FragmentShader.glsl");
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

	View & dCam = CreateDisplay().SetBounds(0.0, 1.0, Attach::Pix(300), 1.0,
			-640.0 / 480).SetHandler(new Handler3D(sCam));

	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(300), true);
	Var<bool> btnReset("UI.Reset System", false, false);
	Var<bool> btnShowKeyFrame("UI.Show Key Frames", false, true);
	Var<bool> btnShowKeyPoint("UI.Show Key Points", true, true);
	Var<bool> btnShowMesh("UI.Show Mesh", false, true);
	Var<bool> btnShowCam("UI.Show Camera", true, true);
	Var<bool> btnFollowCam("UI.Fllow Camera", false, true);
	Var<bool> btnShowNormal("UI.Show Normal", false, true);
	Var<bool> btnShowColor("UI.Show Color Map", false, true);
	Var<bool> btnSaveMesh("UI.Save as Mesh", false, false);
	Var<bool> btnDrawWireFrame("UI.WireFrame Mode", false, true);

	const int width = 640;
	const int height = 480;
	unsigned char* imageArray = new unsigned char[3 * width * height];
	GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB,
			GL_UNSIGNED_BYTE);

	while (1) {

		if (quit)
			std::terminate();

		if (ShouldQuit()) {
			psystem->requestStop = true;
		}

		glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (Pushed(btnReset)) {
			psystem->requestReboot = true;
		}

		if (Pushed(btnSaveMesh)) {
			psystem->requestSaveMesh = true;
		}

		dCam.Activate(sCam);

		if (btnShowKeyFrame)
			drawKeyFrame();

		if (btnShowKeyPoint)
			drawKeys();

		if (btnDrawWireFrame)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

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

		if (btnDrawWireFrame)
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (btnShowCam)
			drawCamera();

		if (btnFollowCam)
			followCam();

		FinishFrame();
	}
}

void Viewer::drawColor() {
	if (mpMap->meshUpdated) {

		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex,
				   sizeof(float3) * mpMap->noTriangles[0] * 3,
				   cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **colorMaped, (void*) mpMap->modelColor,
				   sizeof(uchar3) * mpMap->noTriangles[0] * 3,
				   cudaMemcpyDeviceToDevice);

		mpMap->meshUpdated = false;
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

	if (mpMap->noTrianglesHost == 0)
		return;

	if (mpMap->meshUpdated) {

		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex,
				   sizeof(float3) * mpMap->noTrianglesHost * 3,
				   cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **normalMaped, (void*) mpMap->modelNormal,
				   sizeof(float3) * mpMap->noTrianglesHost * 3,
				   cudaMemcpyDeviceToDevice);

		mpMap->mutexMesh.lock();
		mpMap->meshUpdated = false;
		mpMap->mutexMesh.unlock();
	}

	pangolin::GlSlProgram* program;
	if (bNormal)
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

void Viewer::drawKeyFrame() {
	std::set<KeyFrame *>::iterator iter = mpMap->keyFrames.begin();
	std::set<KeyFrame *>::iterator lend = mpMap->keyFrames.end();
	vector<GLfloat> cam;

	for(; iter != lend; ++iter) {
		Eigen::Vector3f p[4];
		p[0] << 0.1, 0.08, 0;
		p[1] << 0.1, -0.08, 0;
		p[2] << -0.1, 0.08, 0;
		p[3] << -0.1, -0.08, 0;

		Eigen::Matrix3f r = (*iter)->pose.topLeftCorner(3, 3).cast<float>();
		Eigen::Vector3f t = (*iter)->pose.topRightCorner(3, 1).cast<float>();
		for (int i = 0; i < 4; ++i) {
			p[i] = r * p[i] * 0.3 + t;
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

		glColor3f(1.0, 1.0, 1.0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_TRIANGLES, 3);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		cam.clear();
	}
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
	for (int i = 0; i < 5; ++i) {
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
	if (lost)
		glColor3f(1.0, 0.0, 0.0);
	else
		glColor3f(0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_TRIANGLES, 3);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Viewer::drawKeys() {

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

#include "Viewer.h"
#include "KeyFrame.h"

#include <unistd.h>
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <cuda_profiler_api.h>

using namespace std;
using namespace pangolin;

Viewer::Viewer()
: mpMap(nullptr), ptracker(nullptr), psystem(nullptr),
  vao(0), vertexMaped(nullptr), normalMaped(nullptr),
  colorMaped(nullptr), quit(false) {
}

void Viewer::signalQuit() {
	quit = true;
}

void Viewer::spin() {

	CreateWindowAndBind("FUSION", 2560, 1440);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	phongShader.AddShaderFromFile(GlSlVertexShader, "GUI/Shaders/VertexShader.phong.glsl");
	phongShader.AddShaderFromFile(GlSlFragmentShader, "GUI/Shaders/FragmentShader.glsl");
	phongShader.Link();

	normalShader.AddShaderFromFile(GlSlVertexShader, "GUI/Shaders/VertexShader.normal.glsl");
	normalShader.AddShaderFromFile(GlSlFragmentShader, "GUI/Shaders/FragmentShader.glsl");
	normalShader.Link();

	colorShader.AddShaderFromFile(GlSlVertexShader, "GUI/Shaders/VertexShader.color.glsl");
	colorShader.AddShaderFromFile(GlSlFragmentShader, "GUI/Shaders/FragmentShader.glsl");
	colorShader.Link();

	sCam = OpenGlRenderState(
			ProjectionMatrix(640, 480, 520.149963, 516.175781, 309.993548, 227.090932, 0.1f, 1000.0f),
			ModelViewLookAtRUB(0, 0, 0, 0, 0, 1, 0, -1, 0)
	);

	glGenVertexArrays(1, &vao);
	glGenVertexArrays(1, &vao_color);

	vertex.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	vertexMaped = new CudaScopedMappedPtr(vertex);

	normal.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	normalMaped = new CudaScopedMappedPtr(normal);

	color.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	colorMaped = new CudaScopedMappedPtr(color);

	colorImage.Reinitialise(640, 480, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	colorImageMaped = new CudaScopedMappedArray(colorImage);

	depthImage.Reinitialise(640, 480, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	depthImageMaped = new CudaScopedMappedArray(depthImage);

	renderedImage.Reinitialise(640, 480, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	renderedImageMaped = new CudaScopedMappedArray(renderedImage);

	topDownImage.Reinitialise(640, 480, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	topDownImageMaped = new CudaScopedMappedArray(topDownImage);

	View & dCam = CreateDisplay().SetAspect(-640.0 / 480).SetHandler(new Handler3D(sCam));
	View & Image0 = CreateDisplay().SetAspect(-640.0 / 480);
	View & Image1 = CreateDisplay().SetAspect(-640.0 / 480);
	View & Image2 = CreateDisplay().SetAspect(-640.0 / 480);
	View & Image3 = CreateDisplay().SetAspect(-640.0 / 480);
	Display("SubDisplay0").SetBounds(0.0, 1.0,  Attach::Pix(200), 1.0).SetLayout(LayoutOverlay).AddDisplay(Image3).AddDisplay(dCam);
	Display("SubDisplay1").SetBounds(0.0, 1.0, 0.75, 1.0).SetLayout(LayoutEqualVertical).AddDisplay(Image0).AddDisplay(Image1).AddDisplay(Image2);

	CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200), true);
	Var<bool> btnReset("UI.Reset System", false, false);
	Var<bool> btnShowKeyFrame("UI.Show Key Frames", false, true);
	Var<bool> btnShowKeyPoint("UI.Show Key Points", false, true);
	Var<bool> btnShowMesh("UI.Show Mesh", true, true);
	Var<bool> btnShowCam("UI.Show Camera", true, true);
	Var<bool> btnFollowCam("UI.Fllow Camera", false, true);
	Var<bool> btnShowNormal("UI.Show Normal", false, true);
	Var<bool> btnShowColor("UI.Show Color Map", false, true);
	Var<bool> btnSaveMesh("UI.Save as Mesh", false, false);
	Var<bool> btnDrawWireFrame("UI.WireFrame Mode", false, true);
	Var<bool> btnShowColorImage("UI.Color Image", true, true);
	Var<bool> btnShowDepthImage("UI.Depth Image", false, true);
	Var<bool> btnShowRenderedImage("UI.Rendered Image", false, true);
	Var<bool> btnPauseSystem("UI.Pause System", false, false);
	Var<bool> btnUseGraphMatching("UI.Graph Matching", false, true);
	Var<bool> btnLocalisationMode("UI.Localisation Only", false, true);
	Var<bool> btnShowTopDownView("UI.Top Down View", false, true);

	while (1) {

		if (quit) {
			std::terminate();
		}

		if (ShouldQuit()) {
			SafeCall(cudaProfilerStop());
			psystem->requestStop = true;
		}

		glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (Pushed(btnReset)) {
			btnLocalisationMode = false;
			psystem->requestReboot = true;
		}

		if (Pushed(btnSaveMesh))
			psystem->requestSaveMesh = true;

		if (btnUseGraphMatching) {
			if(!ptracker->graphMatching)
				ptracker->graphMatching = true;
		}
		else {
			if(ptracker->graphMatching)
				ptracker->graphMatching = false;
		}

		if (btnLocalisationMode) {
			if(!ptracker->localisationOnly)
				ptracker->localisationOnly = true;
		}
		else {
			if(ptracker->localisationOnly)
				ptracker->localisationOnly = false;
		}

		dCam.Activate(sCam);

		if (btnShowMesh || btnShowNormal || btnShowColor)
			psystem->requestMesh = true;
		else
			psystem->requestMesh = false;

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

		if (btnShowRenderedImage ||
			btnShowDepthImage ||
			btnShowColorImage) {
			if(!ptracker->needImages)
				ptracker->needImages = true;
		}
		else
			if(ptracker->needImages)
				ptracker->needImages = false;

		Image0.Activate();
		if (btnShowRenderedImage)
			showPrediction();

		Image1.Activate();
		if (btnShowDepthImage)
			showDepthImage();

		Image2.Activate();
		if (btnShowColorImage)
			showColorImage();

		Image3.Activate();
		if (btnShowTopDownView)
			topDownView();

		if (ptracker->imageUpdated)
			ptracker->imageUpdated = false;

		FinishFrame();
	}
}

void Viewer::topDownView() {
	if(psystem->imageUpdated) {
		SafeCall(cudaMemcpy2DToArray(**topDownImageMaped, 0, 0,
				(void*) psystem->renderedImage.data,
				psystem->renderedImage.step, sizeof(uchar4) * 640, 480,
				cudaMemcpyDeviceToDevice));
		psystem->imageUpdated = false;
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	topDownImage.RenderToViewport(true);
}

void Viewer::showPrediction() {
	if(ptracker->imageUpdated) {
		SafeCall(cudaMemcpy2DToArray(**renderedImageMaped, 0, 0,
				(void*) ptracker->renderedImage.data,
				 ptracker->renderedImage.step, sizeof(uchar4) * 640, 480,
				 cudaMemcpyDeviceToDevice));
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	renderedImage.RenderToViewport(true);
}

void Viewer::showDepthImage() {
	if(ptracker->imageUpdated) {
		SafeCall(cudaMemcpy2DToArray(**depthImageMaped, 0, 0,
				(void*) ptracker->renderedDepth.data,
				 ptracker->renderedDepth.step, sizeof(uchar4) * 640, 480,
				 cudaMemcpyDeviceToDevice));
	}

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	depthImage.RenderToViewport(true);
}

void Viewer::showColorImage() {
	if(ptracker->imageUpdated) {
		SafeCall(cudaMemcpy2DToArray(**colorImageMaped, 0, 0,
				(void*) ptracker->rgbaImage.data,
				ptracker->rgbaImage.step, sizeof(uchar4) * 640, 480,
				cudaMemcpyDeviceToDevice));
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	colorImage.RenderToViewport(true);
}

void Viewer::drawColor() {
	if (mpMap->meshUpdated) {
		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex, sizeof(float3) * mpMap->noTrianglesHost * 3,  cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **colorMaped, (void*) mpMap->modelColor, sizeof(uchar3) * mpMap->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		mpMap->meshUpdated = false;
	}

	colorShader.SaveBind();
	colorShader.SetUniform("viewMat", sCam.GetModelViewMatrix());
	colorShader.SetUniform("projMat", sCam.GetProjectionMatrix());
	glBindVertexArray(vao_color);
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
	sCam.SetModelViewMatrix(ModelViewLookAtRUB(eye(0), eye(1), eye(2), look(0), look(1), look(2), up(0), up(1), up(2)));
}

void Viewer::drawMesh(bool bNormal) {

	if (mpMap->noTrianglesHost == 0)
		return;

	if (mpMap->meshUpdated) {
		cudaMemcpy((void*) **vertexMaped, (void*) mpMap->modelVertex, sizeof(float3) * mpMap->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **normalMaped, (void*) mpMap->modelNormal, sizeof(float3) * mpMap->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		mpMap->meshUpdated = false;
	}

	pangolin::GlSlProgram * program;
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
	vector<GLfloat> points;
	std::set<KeyFrame *>::iterator iter = mpMap->keyFrames.begin();
	std::set<KeyFrame *>::iterator lend = mpMap->keyFrames.end();

	for(; iter != lend; ++iter) {
		Eigen::Vector3d trans = (*iter)->translation();
		points.push_back(trans(0));
		points.push_back(trans(1));
		points.push_back(trans(2));
	}

	glColor3f(1.0, 0.0, 0.0);
	glPointSize(3.0);
	glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_POINTS, 3);
	glPointSize(1.0);
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
		p[i] = rotation * p[i] * 0.5 + translation;
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
	std::vector<ORBKey> allKeys = mpMap->getAllKeys();
	if(allKeys.size() == 0)
		return;

	vector<GLfloat> points;
	for(int i = 0; i < allKeys.size(); ++i) {
		points.push_back(allKeys[i].pos.x);
		points.push_back(allKeys[i].pos.y);
		points.push_back(allKeys[i].pos.z);
	}

	glColor3f(1.0, 0.0, 0.0);
	glPointSize(3.0);
	glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_POINTS, 3);
	glPointSize(1.0);
}

void Viewer::setMap(Mapping* pMap) {
	mpMap = pMap;
}

void Viewer::setSystem(System* pSystem) {
	psystem = pSystem;
}

void Viewer::setTracker(Tracker* pTracker) {
	ptracker = pTracker;
}

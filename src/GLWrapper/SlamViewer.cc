
#include "SlamViewer.h"
#include "KeyFrame.h"
#include <unistd.h>
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <cuda_profiler_api.h>
using namespace std;
using namespace pangolin;

SlamViewer::SlamViewer() :
		map(0), tracker(0), system(0), vertexArrayObjectMesh(0), meshVerticesCUDAMapped(0),
		meshNormalsCUDAMapped(0), meshTextureCUDAMapped(0), quit(false) {
}

SlamViewer::SlamViewer(int w, int h, Eigen::Matrix3f K, std::string title) :
		imageDepthCUDAMapped(0), imageSyntheticViewCUDAMapped(0), map(0),
		imageTextureCUDAMapped(0), imageTopdownViewCUDAMapped(0), tracker(0),
		system(0), vertexArrayObjectMesh(0), quit(false), bufferSizeImage(0), bufferSizeTriangles(0),
		bufferSizeVertices(0)
{
	// Create GUI window
	int window_width = w * 4;
	int window_height= h * 3;
	CreateWindowAndBind(title, window_width, window_height);

	// Make sure we Enable a bunch of gl state
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Load shading programs and link
	shaderPhong.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.phong.glsl");
	shaderPhong.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderPhong.Link();

	shaderNormalmap.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.normal.glsl");
	shaderNormalmap.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderNormalmap.Link();

	shaderTexture.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.color.glsl");
	shaderTexture.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderTexture.Link();

	// Create viewing camera
	viewCam = OpenGlRenderState(
			ProjectionMatrix(w, h, K(0,0), K(1,1), K(0,2), K(1,2), 0.1f, 100.0f),
			ModelViewLookAtRUB(0, 0, 0, 0, 0, 1, 0, -1, 0)
	);

	// Create vertex arrays for storing mesh
	glGenVertexArrays(1, &vertexArrayObjectMesh);
	glGenVertexArrays(1, &vertexArrayObjectColor);

	// Map OpenGL buffer array to cuda
	meshVertices.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshVerticesCUDAMapped = new CudaScopedMappedPtr(meshVertices);

	meshNormals.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshNormalsCUDAMapped = new CudaScopedMappedPtr(meshNormals);

	meshTexture.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshTextureCUDAMapped = new CudaScopedMappedPtr(meshTexture);
}

void SlamViewer::signalQuit() {
	quit = true;
}

void SlamViewer::spin() {

	CreateWindowAndBind("FUSION", 2560, 1440);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	shaderPhong.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.phong.glsl");
	shaderPhong.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderPhong.Link();

	shaderNormalmap.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.normal.glsl");
	shaderNormalmap.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderNormalmap.Link();

	shaderTexture.AddShaderFromFile(GlSlVertexShader, "src/GLWrapper/OpenGL/VertexShader.color.glsl");
	shaderTexture.AddShaderFromFile(GlSlFragmentShader, "src/GLWrapper/OpenGL/FragmentShader.glsl");
	shaderTexture.Link();

	viewCam = OpenGlRenderState(
			ProjectionMatrix(640, 480, 520.149963, 516.175781, 309.993548, 227.090932, 0.1f, 1000.0f),
			ModelViewLookAtRUB(0, 0, 0, 0, 0, 1, 0, -1, 0)
	);

	glGenVertexArrays(1, &vertexArrayObjectMesh);
	glGenVertexArrays(1, &vertexArrayObjectColor);

	meshVertices.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshVerticesCUDAMapped = new CudaScopedMappedPtr(meshVertices);

	meshNormals.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshNormalsCUDAMapped = new CudaScopedMappedPtr(meshNormals);

	meshTexture.Reinitialise(GlArrayBuffer, DeviceMap::MaxVertices,
	GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
	meshTextureCUDAMapped = new CudaScopedMappedPtr(meshTexture);

	imageTexture.Reinitialise(640, 480, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	imageTextureCUDAMapped = new CudaScopedMappedArray(imageTexture);

	imageDepth.Reinitialise(640, 480, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	imageDepthCUDAMapped = new CudaScopedMappedArray(imageDepth);

	imageSyntheticView.Reinitialise(640, 480, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	imageSyntheticViewCUDAMapped = new CudaScopedMappedArray(imageSyntheticView);

	imageTopdownView.Reinitialise(640, 480, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	imageTopdownViewCUDAMapped = new CudaScopedMappedArray(imageTopdownView);

	View & dCam = CreateDisplay().SetAspect(-640.0 / 480).SetHandler(new Handler3D(viewCam));
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
	Var<bool> btnShowDepthImage("UI.Depth Image", true, true);
	Var<bool> btnShowRenderedImage("UI.Rendered Image", true, true);
	Var<bool> btnPauseSystem("UI.Pause System", false, false);
	Var<bool> btnUseGraphMatching("UI.Graph Matching", true, true);
	Var<bool> btnLocalisationMode("UI.Localisation Only", false, true);
	Var<bool> btnShowTopDownView("UI.Top Down View", false, true);
	Var<bool> btnWriteMapToDisk("UI.Write Map to Disk", false, false);
	Var<bool> btnReadMapFromDisk("UI.Read Map From Disk", false, false);

	while (1) {

		if (quit) {
			std::terminate();
		}

		if (ShouldQuit()) {
			SafeCall(cudaProfilerStop());
			system->requestStop = true;
		}

		glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (Pushed(btnReset)) {
			btnLocalisationMode = false;
			system->requestReboot = true;
		}

		if (Pushed(btnSaveMesh))
			system->requestSaveMesh = true;

		if (btnUseGraphMatching) {
			if(!tracker->useGraphMatching)
				tracker->useGraphMatching = true;
		}
		else {
			if(tracker->useGraphMatching)
				tracker->useGraphMatching = false;
		}

		if (Pushed(btnWriteMapToDisk)) {
			system->requestSaveMap = true;
		}

		if (Pushed(btnReadMapFromDisk)) {
			system->requestReadMap = true;
			btnLocalisationMode = true;
		}

		if (btnLocalisationMode) {
			if(!tracker->mappingDisabled)
				tracker->mappingDisabled = true;
		}
		else {
			if(tracker->mappingDisabled)
				tracker->mappingDisabled = false;
		}

		dCam.Activate(viewCam);

		if (btnShowMesh || btnShowNormal || btnShowColor)
			system->requestMesh = true;
		else
			system->requestMesh = false;

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
			if(!tracker->needImages)
				tracker->needImages = true;
		}
		else
			if(tracker->needImages)
				tracker->needImages = false;

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

		if (tracker->imageUpdated)
			tracker->imageUpdated = false;

		FinishFrame();
	}
}

void SlamViewer::topDownView() {
	if(system->imageUpdated) {
		SafeCall(cudaMemcpy2DToArray(**imageTopdownViewCUDAMapped, 0, 0,
				(void*) system->renderedImage.data,
				system->renderedImage.step, sizeof(uchar4) * 640, 480,
				cudaMemcpyDeviceToDevice));
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageTopdownView.RenderToViewport(true);
}

void SlamViewer::showPrediction() {
	if(tracker->imageUpdated) {
		if(tracker->updateImageMutex.try_lock()) {
			SafeCall(cudaMemcpy2DToArray(**imageSyntheticViewCUDAMapped, 0, 0,
					(void*) tracker->renderedImage.data,
					 tracker->renderedImage.step, sizeof(uchar4) * 640, 480,
					 cudaMemcpyDeviceToDevice));
			tracker->updateImageMutex.unlock();
		}
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageSyntheticView.RenderToViewport(true);
}

void SlamViewer::showDepthImage() {
	if(tracker->imageUpdated) {
		if(tracker->updateImageMutex.try_lock()) {
			SafeCall(cudaMemcpy2DToArray(**imageDepthCUDAMapped, 0, 0,
					(void*) tracker->renderedDepth.data,
					 tracker->renderedDepth.step, sizeof(uchar4) * 640, 480,
					 cudaMemcpyDeviceToDevice));
			tracker->updateImageMutex.unlock();
		}
	}

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageDepth.RenderToViewport(true);
}

void SlamViewer::showColorImage() {
	if(tracker->imageUpdated) {
		if(tracker->updateImageMutex.try_lock()) {
			SafeCall(cudaMemcpy2DToArray(**imageTextureCUDAMapped, 0, 0,
					(void*) tracker->rgbaImage.data,
					tracker->rgbaImage.step, sizeof(uchar4) * 640, 480,
					cudaMemcpyDeviceToDevice));
			tracker->updateImageMutex.unlock();
		}
	}
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageTexture.RenderToViewport(true);
}

void SlamViewer::drawColor() {
	if (map->meshUpdated) {
		cudaMemcpy((void*) **meshVerticesCUDAMapped, (void*) map->modelVertex, sizeof(float3) * map->noTrianglesHost * 3,  cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshNormalsCUDAMapped, (void*) map->modelNormal, sizeof(float3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshTextureCUDAMapped, (void*) map->modelColor, sizeof(uchar3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		map->meshUpdated = false;
	}

	shaderTexture.SaveBind();
	shaderTexture.SetUniform("viewMat", viewCam.GetModelViewMatrix());
	shaderTexture.SetUniform("projMat", viewCam.GetProjectionMatrix());
	glBindVertexArray(vertexArrayObjectColor);
	meshVertices.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	meshVertices.Unbind();

	meshTexture.Bind();
	glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	meshTexture.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, map->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	shaderTexture.Unbind();
	glBindVertexArray(0);
}

void SlamViewer::followCam() {

	Eigen::Matrix4f pose = tracker->GetCurrentPose();
	Eigen::Matrix3f rotation = pose.topLeftCorner(3, 3);
	Eigen::Vector3f translation = pose.topRightCorner(3, 1);
	Eigen::Vector3f up = { 0, -1, 0 };
	Eigen::Vector3f eye = { 0, 0, 0 };
	Eigen::Vector3f look = { 0, 0, 1 };
	up = rotation * up;
	eye = rotation * eye + translation;
	look = rotation * look + translation;
	viewCam.SetModelViewMatrix
	(
		ModelViewLookAtRUB( eye(0),  eye(1),  eye(2),
						   look(0), look(1), look(2),
						     up(0),   up(1),   up(2))
	);
}

void SlamViewer::drawMesh(bool bNormal) {

	if (map->noTrianglesHost == 0)
		return;

	if (map->meshUpdated) {
		cudaMemcpy((void*) **meshVerticesCUDAMapped, (void*) map->modelVertex, sizeof(float3) * map->noTrianglesHost * 3,  cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshNormalsCUDAMapped, (void*) map->modelNormal, sizeof(float3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshTextureCUDAMapped, (void*) map->modelColor, sizeof(uchar3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		map->meshUpdated = false;
	}

	pangolin::GlSlProgram * program;
	if (bNormal)
		program = &shaderNormalmap;
	else
		program = &shaderPhong;

	program->SaveBind();
	program->SetUniform("viewMat", viewCam.GetModelViewMatrix());
	program->SetUniform("projMat", viewCam.GetProjectionMatrix());

	glBindVertexArray(vertexArrayObjectMesh);
	meshVertices.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	meshVertices.Unbind();

	meshNormals.Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, 0);
	glEnableVertexAttribArray(1);
	meshNormals.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, map->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	program->Unbind();
	glBindVertexArray(0);
}

void SlamViewer::Insert(std::vector<GLfloat>& vPt, Eigen::Vector3f& pt) {

	vPt.push_back(pt(0));
	vPt.push_back(pt(1));
	vPt.push_back(pt(2));
}

void SlamViewer::drawKeyFrame() {

	vector<GLfloat> points;
	std::vector<KeyFrame *> allKFs = map->GlobalMap();
	std::sort(allKFs.begin(), allKFs.end(), [](KeyFrame * a, KeyFrame * b) {return a->frameId < b->frameId;});

	for (KeyFrame * kf : allKFs)
	{
		Eigen::Matrix3f rot = kf->Rotation();
		Eigen::Vector3f trans = kf->Translation();
		points.clear();
		for(Eigen::Vector3f point : camVertices)
		{
			point = rot * point + trans;
			points.push_back(point(0));
			points.push_back(point(1));
			points.push_back(point(2));
		}

		glColor3f(0.5f, 0.3f, 0.5f);
		glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_LINE_STRIP, 3);
	}

	points.clear();
	for (KeyFrame * kf : allKFs)
	{
		Eigen::Vector3f trans = kf->pose.topRightCorner(3, 1);
		points.push_back(trans(0));
		points.push_back(trans(1));
		points.push_back(trans(2));
	}

	glColor3f(0.0, 0.5, 0.6);
	glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_LINE_STRIP, 3);
}

void SlamViewer::drawCamera() {

	vector<GLfloat> cam;
	Eigen::Vector3f p[5];
	p[0] << 0.1, 0.08, 0;
	p[1] << 0.1, -0.08, 0;
	p[2] << -0.1, 0.08, 0;
	p[3] << -0.1, -0.08, 0;
	p[4] << 0, 0, -0.08;

	Eigen::Matrix4f pose = tracker->GetCurrentPose();
	Eigen::Matrix3f rotation = pose.topLeftCorner(3, 3);
	Eigen::Vector3f translation = pose.topRightCorner(3, 1);
	for (int i = 0; i < 5; ++i) {
		p[i] = rotation * p[i] * 0.5 + translation;
	}

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

	bool lost = (tracker->state == -1);
	if (lost)
		glColor3f(1.0, 0.0, 0.0);
	else {
		glColor3f(0.0, 1.0, 0.0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_TRIANGLES, 3);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void SlamViewer::drawKeys() {

	if(map->noKeysHost == 0)
		return;

	vector<GLfloat> points;
	for(int i = 0; i < map->noKeysHost; ++i) {
		points.push_back(map->hostKeys[i].pos.x);
		points.push_back(map->hostKeys[i].pos.y);
		points.push_back(map->hostKeys[i].pos.z);
	}

//	std::vector<KeyFrame *> KFs = map->GlobalMap();
//	for(int i = 0; i < KFs.size(); ++i) {
//		KeyFrame * kf = KFs[i];
//		for(int j = 0; j < kf->pt3d.size(); ++j) {
//			MapPoint * mp = kf->pt3d[j];
//			if(mp && mp->observations.size() > 1) {
//				points.push_back(mp->GetWorldPosition()(0));
//				points.push_back(mp->GetWorldPosition()(1));
//				points.push_back(mp->GetWorldPosition()(2));
//			}
//		}
//	}

	glColor3f(1.0, 0.0, 0.0);
	glPointSize(3.0);
	glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_POINTS, 3);
	glPointSize(1.0);

	points.clear();
	for (int i = 0; i < tracker->output.size(); ++i) {
		points.push_back(tracker->output[i](0));
		points.push_back(tracker->output[i](1));
		points.push_back(tracker->output[i](2));
	}

	glColor3f(0.0, 1.0, 0.0);
	glPointSize(10.0);
	glDrawVertices(points.size() / 3, (GLfloat*) &points[0], GL_POINTS, 3);
	glPointSize(1.0);
}

void SlamViewer::setMap(DistanceField* pMap) {
	map = pMap;
}

void SlamViewer::setSystem(System* pSystem) {
	system = pSystem;
}

void SlamViewer::setTracker(Tracker* pTracker) {
	tracker = pTracker;
}

std::vector<GLfloat> SlamViewer::getTransformedCamera(SE3 pose) const
{
	std::vector<GLfloat> result;
	for (Eigen::Vector3f vertex : camVertices)
	{
		Eigen::Vector3f vertex_transformed = pose.rotationMatrix().cast<float>() * vertex + pose.translation().cast<float>();
		result.push_back(vertex_transformed(0));
		result.push_back(vertex_transformed(1));
		result.push_back(vertex_transformed(2));
	}
	return result;
}

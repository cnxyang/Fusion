#include "KeyFrame.h"
#include "GlViewer.h"
#include "Legacy/Tracking.h"

#include <unistd.h>
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

using namespace std;
using namespace pangolin;

void GlViewer::processMessages()
{
	// if SYSTEM RESET
	if (Pushed(*buttonSystemReset))
	{
		*buttonToggleLocalisationMode = false;
		system->requestReboot = true;
	}

	// if EXPORT MESH TO FILE
	if (Pushed(*buttonExportMeshToFile))
		system->requestSaveMesh = true;

	// if WRITE MAP TO BINARY FILE
	if (Pushed(*buttonWriteMapToDiskBinary))
	{
		system->requestSaveMap = true;
	}

	// if READ MAP FROM BINARY FILE
	if (Pushed(*buttonReadMapFromDiskBinary))
	{
		system->requestReadMap = true;
		*buttonToggleLocalisationMode = true;
	}
}

void GlViewer::drawViewsToScreen()
{
	glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	modelViewCamera.Activate(viewCam);

	if (*buttonRenderSceneMesh ||
		*buttonRenderSceneNormal ||
		*buttonRenderSceneRGB)
	{
		system->requestMesh = true;
	}
	else
	{
		system->requestMesh = false;
	}

	if (*buttonShowPoseGraph)
		drawKeyFrameGraph();

	if (*buttonShowKeyPoints)
		drawKeys();

	if (*buttonToggleWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	if (*buttonRenderSceneMesh)
	{
		if (*buttonRenderSceneNormal)
			*buttonRenderSceneNormal = false;
		if (*buttonRenderSceneRGB)
			*buttonRenderSceneRGB = false;
		drawMesh(false);
	}

	if (*buttonRenderSceneNormal)
	{
		if (*buttonRenderSceneRGB)
			*buttonRenderSceneRGB = false;
		drawMesh(true);
	}

	if (*buttonRenderSceneRGB)
	{
		drawColor();
	}

	if (*buttonToggleWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (*buttonShowCurrentCamera)
		drawCurrentCamera();

	if (*buttonFollowCamera)
		setModelViewFollowCamera();

	if (buttonEnableSyntheticView->Get())
	{
		imageSyntheticView.Activate();
		drawSyntheticViewToCamera();
	}

	if (*buttonEnableDepthImage)
	{
		imageDepthView.Activate();
		drawDepthViewToCamera();
	}

	if (*buttonEnableRGBImage)
	{
		imageRGBView.Activate();
		drawRGBViewToCamera();
	}

	if (*buttonEnableBirdsEyeView)
	{
		imageBirdsEyeView.Activate();
		drawBirdsEyeViewToCamera();
	}

	FinishFrame();

	if(pangolin::ShouldQuit())
		return;
}

void GlViewer::drawBirdsEyeViewToCamera() {
//	if(system->imageUpdated) {
//		SafeCall(cudaMemcpy2DToArray(**imageBirdsEyeCUDAMapped, 0, 0,
//				(void*) system->renderedImage.data,
//				system->renderedImage.step, sizeof(uchar4) * 640, 480,
//				cudaMemcpyDeviceToDevice));
//	}
//	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//	imageBirdsEye.RenderToViewport(true);
}

void GlViewer::drawSyntheticViewToCamera() {
//	if(tracker->imageUpdated) {
//		if(tracker->updateImageMutex.try_lock()) {
//			SafeCall(cudaMemcpy2DToArray(**imageSyntheticCUDAMapped, 0, 0,
//					(void*) tracker->renderedImage.data,
//					 tracker->renderedImage.step, sizeof(uchar4) * 640, 480,
//					 cudaMemcpyDeviceToDevice));
//			tracker->updateImageMutex.unlock();
//		}
//	}
//	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//	imageSynthetic.RenderToViewport(true);
}

void GlViewer::drawDepthViewToCamera() {
//	if(tracker->imageUpdated) {
//		if(tracker->updateImageMutex.try_lock()) {
//			SafeCall(cudaMemcpy2DToArray(**imageDepthCUDAMapped, 0, 0,
//					(void*) tracker->renderedDepth.data,
//					 tracker->renderedDepth.step, sizeof(uchar4) * 640, 480,
//					 cudaMemcpyDeviceToDevice));
//			tracker->updateImageMutex.unlock();
//		}
//	}
//
//	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//	imageDepth.RenderToViewport(true);
}

void GlViewer::drawRGBViewToCamera() {
//	if(tracker->imageUpdated) {
//		if(tracker->updateImageMutex.try_lock()) {
//			SafeCall(cudaMemcpy2DToArray(**imageRGBCUDAMapped, 0, 0,
//					(void*) tracker->rgbaImage.data,
//					tracker->rgbaImage.step, sizeof(uchar4) * 640, 480,
//					cudaMemcpyDeviceToDevice));
//			tracker->updateImageMutex.unlock();
//		}
//	}
//	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//	imageRGB.RenderToViewport(true);
}

void GlViewer::drawColor() {
	if (map->meshUpdated) {
		cudaMemcpy((void*) **meshVerticesCUDAMapped, (void*) map->modelVertex, sizeof(float3) * map->noTrianglesHost * 3,  cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshNormalsCUDAMapped, (void*) map->modelNormal, sizeof(float3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshTextureCUDAMapped, (void*) map->modelColor, sizeof(uchar3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		map->meshUpdated = false;
	}

	shaderTexture.SaveBind();
	shaderTexture.SetUniform("viewMat", viewCam.GetModelViewMatrix());
	shaderTexture.SetUniform("projMat", viewCam.GetProjectionMatrix());
	glBindVertexArray(vaoFULL);
	bufferVertices.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	bufferVertices.Unbind();

	bufferTexture.Bind();
	glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	bufferTexture.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, map->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	shaderTexture.Unbind();
	glBindVertexArray(0);
}


void GlViewer::drawMesh(bool bNormal) {

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
		program = &shaderNormal;
	else
		program = &shaderPhong;

	program->SaveBind();
	program->SetUniform("viewMat", viewCam.GetModelViewMatrix());
	program->SetUniform("projMat", viewCam.GetProjectionMatrix());

	glBindVertexArray(vaoVerticesAndNormal);
	bufferVertices.Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	bufferVertices.Unbind();

	bufferNormals.Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, 0);
	glEnableVertexAttribArray(1);
	bufferNormals.Unbind();

	glDrawArrays(GL_TRIANGLES, 0, map->noTrianglesHost * 3);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	program->Unbind();
	glBindVertexArray(0);
}



void GlViewer::drawKeys() {

}

void GlViewer::setMap(DenseMap* pMap) {
	map = pMap;
}

void GlViewer::setSystem(SlamSystem* pSystem) {
	system = pSystem;
}
//
//void GlViewer::setTracker(Tracker* pTracker) {
//	tracker = pTracker;
//}

// ===================== REFACTORING ==============================

GlViewer::GlViewer(std::string title, int w, int h, Eigen::Matrix3f K) :
		windowTitle(title), map(NULL), bufferSizeImage(0),
		bufferSizeVertices(0), system(NULL), bufferSizeTriangles(0)
{
	// create a OpenGL context and bind to current thread
	pangolin::CreateWindowAndBind(windowTitle, 2560, 1440);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// TODO: these files paths should be in a config file
	std::string phongShaderPath = "src/GlWrapper/OpenGL/VertexShader.phong.glsl";
	std::string normalShaderPath = "src/GlWrapper/OpenGL/VertexShader.normal.glsl";
	std::string rgbShaderPath = "src/GlWrapper/OpenGL/VertexShader.color.glsl";
	std::string fragmentShaderPath = "src/GlWrapper/OpenGL/FragmentShader.glsl";

	// Load and compile phong shader
	shaderPhong.AddShaderFromFile(GlSlVertexShader, phongShaderPath);
	shaderPhong.AddShaderFromFile(GlSlFragmentShader, fragmentShaderPath);
	shaderPhong.Link();

	// Load and compile normal shader
	shaderNormal.AddShaderFromFile(GlSlVertexShader, normalShaderPath);
	shaderNormal.AddShaderFromFile(GlSlFragmentShader, fragmentShaderPath);
	shaderNormal.Link();

	// Load and compile texture shader
	shaderTexture.AddShaderFromFile(GlSlVertexShader, rgbShaderPath);
	shaderTexture.AddShaderFromFile(GlSlFragmentShader, fragmentShaderPath);
	shaderTexture.Link();

	// Main camera settings
	viewCam = pangolin::OpenGlRenderState(
			pangolin::ProjectionMatrix(w, h, K(0, 0), K(1, 1), K(0, 2), K(1, 2), 0.1f, 100.0f),
			pangolin::ModelViewLookAtRUB(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f)
	);

	// Generate array object for rendering
	glGenVertexArrays(1, &vaoVerticesAndNormal);
	glGenVertexArrays(1, &vaoFULL);

	// Initialise vertex array for shaded mesh
	bufferVertices.Reinitialise(
			GlArrayBuffer,
			DeviceMap::MaxVertices,
			GL_FLOAT, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Initialise vertex array for coloured normal
	bufferNormals.Reinitialise(
			GlArrayBuffer,
			DeviceMap::MaxVertices,
			GL_FLOAT, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Initialise vertex array for shaded rgb
	bufferTexture.Reinitialise(
			GlArrayBuffer,
			DeviceMap::MaxVertices,
			GL_UNSIGNED_BYTE, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Bind vertex array to CUDA
	meshVerticesCUDAMapped = new CudaScopedMappedPtr(bufferVertices);
	meshNormalsCUDAMapped = new CudaScopedMappedPtr(bufferNormals);
	meshTextureCUDAMapped = new CudaScopedMappedPtr(bufferTexture);

	// Initialise texture array
	imageRGB.Reinitialise(w, h, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	imageDepth.Reinitialise(w, h, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	imageSynthetic.Reinitialise(w, h, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	imageBirdsEye.Reinitialise(w, h, GL_RGBA, true, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// bind texture array to CUDA
	imageRGBCUDAMapped = new CudaScopedMappedArray(imageRGB);
	imageDepthCUDAMapped = new CudaScopedMappedArray(imageDepth);
	imageSyntheticCUDAMapped = new CudaScopedMappedArray(imageSynthetic);
	imageBirdsEyeCUDAMapped = new CudaScopedMappedArray(imageBirdsEye);

	modelViewCamera = CreateDisplay().SetAspect(-640.0 / 480).SetHandler(new Handler3D(viewCam));
	imageSyntheticView = CreateDisplay().SetAspect(-640.0 / 480);
	imageDepthView = CreateDisplay().SetAspect(-640.0 / 480);
	imageRGBView = CreateDisplay().SetAspect(-640.0 / 480);
	imageBirdsEyeView = CreateDisplay().SetAspect(-640.0 / 480);

	Display("SubDisplay0").SetBounds(0.0, 1.0,  Attach::Pix(200), 1.0).
			SetLayout(LayoutOverlay).
			AddDisplay(imageBirdsEyeView).
			AddDisplay(modelViewCamera);
	Display("SubDisplay1").SetBounds(0.0, 1.0, 0.75, 1.0).
			SetLayout(LayoutEqualVertical).
			AddDisplay(imageSyntheticView).
			AddDisplay(imageDepthView).
			AddDisplay(imageRGBView);

	// create menu entry i.e. a bunch of buttons
	panelMainMenu = CreatePanel("UI").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200), true);
	buttonSystemReset = new Var<bool>("UI.System Reset", false, false);
	buttonShowPoseGraph = new Var<bool>("UI.Show Pose Graph", false, true);
	buttonShowKeyPoints = new Var<bool>("UI.Show Key Points", false, true);
	buttonRenderSceneMesh = new Var<bool>("UI.Toggle Scene Mesh", true, true);
	buttonShowCurrentCamera = new Var<bool>("UI.Toggle Current Camera", true, true);
	buttonFollowCamera = new Var<bool>("UI.Fllow Camera", false, true);
	buttonRenderSceneNormal = new Var<bool>("UI.Show Normal", false, true);
	buttonRenderSceneRGB = new Var<bool>("UI.Show Color Map", false, true);
	buttonExportMeshToFile = new Var<bool>("UI.Export to File", false, false);
	buttonToggleWireFrame = new Var<bool>("UI.WireFrame Mode", false, true);
	buttonEnableRGBImage = new Var<bool>("UI.Color Image", true, true);
	buttonEnableDepthImage = new Var<bool>("UI.Depth Image", true, true);
	buttonEnableSyntheticView = new Var<bool>("UI.Rendered Image", true, true);
	buttonPauseSystem = new Var<bool>("UI.Pause System", false, false);
	buttonEnableGraphMatching = new Var<bool>("UI.Graph Matching", true, true);
	buttonToggleLocalisationMode = new Var<bool>("UI.Localisation Only", false, true);
	buttonEnableBirdsEyeView = new Var<bool>("UI.Top Down View", false, true);
	buttonWriteMapToDiskBinary = new Var<bool>("UI.Write Map to Disk", false, false);
	buttonReadMapFromDiskBinary = new Var<bool>("UI.Read Map From Disk", false, false);

	// unbind current context from the main thread
	pangolin::GetBoundWindow()->RemoveCurrent();
}

GlViewer::~GlViewer()
{
	delete buttonSystemReset;
	delete buttonShowPoseGraph;
	delete buttonShowKeyPoints;
	delete buttonRenderSceneMesh;
	delete buttonShowCurrentCamera;
	delete buttonFollowCamera;
	delete buttonRenderSceneNormal;
	delete buttonRenderSceneRGB;
	delete buttonExportMeshToFile;
	delete buttonToggleWireFrame;
	delete buttonEnableRGBImage;
	delete buttonEnableDepthImage;
	delete buttonEnableSyntheticView;
	delete buttonPauseSystem;
	delete buttonEnableGraphMatching;
	delete buttonToggleLocalisationMode;
	delete buttonEnableBirdsEyeView;
	delete buttonWriteMapToDiskBinary;
	delete buttonReadMapFromDiskBinary;

	pangolin::GetBoundWindow()->RemoveCurrent();
	pangolin::DestroyWindow(windowTitle);
}

void GlViewer::setCurrentImages(PointCloud* data)
{

}

void GlViewer::setModelViewFollowCamera()
{
	Eigen::Vector3d up = { 0, -1, 0 }, eye = { 0, 0, 0 }, look = { 0, 0, 1 };
	// up vector is the up direction of the camera
	up = currentCamPose.rotationMatrix() * up;
	// eye point which happens to be the translational part of the camera pose
	eye = currentCamPose.rotationMatrix() * eye + currentCamPose.translation();
	// looking at : NOTE this is a point in the world coordinate rather than a vector
	look = currentCamPose.rotationMatrix() * look + currentCamPose.translation();

	// set model view matrix ( eye, look, up ) OpenGl style;
	viewCam.SetModelViewMatrix
	(
		ModelViewLookAtRUB( eye(0),  eye(1),  eye(2),
						   look(0), look(1), look(2),
						     up(0),   up(1),   up(2))
	);
}

void GlViewer::drawCurrentCamera() const {

	std::vector<GLfloat> cam = getTransformedCam(currentCamPose);
	glColor3fv(RGBActiveCam);
	glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_LINE_STRIP, 3);
}

void GlViewer::drawKeyFrameGraph() const {

	for (SE3 pose : keyFrameGraph)
	{
		std::vector<GLfloat> cam = getTransformedCam(pose);
		glColor3fv(RGBKeyFrameGraph);
		glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_LINE_STRIP, 3);
	}
}

std::vector<GLfloat> GlViewer::getTransformedCam(SE3 pose) const
{
	std::vector<GLfloat> result;

	// Generate a transformed array of points
	// which represents the camera wire-frame
	for (Eigen::Vector3f vertex : camVertices)
	{
		Eigen::Vector3f vertex_transformed = pose.rotationMatrix().cast<float>() * vertex + pose.translation().cast<float>();
		result.push_back(vertex_transformed(0));
		result.push_back(vertex_transformed(1));
		result.push_back(vertex_transformed(2));
	}

	return result;
}

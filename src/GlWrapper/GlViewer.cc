#include "KeyFrame.h"
#include "GlViewer.h"
#include "VoxelMap.h"
#include "PointCloud.h"
#include <unistd.h>
#include <algorithm>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

#define PushButton(name) Var<bool>(name, false, false)
#define CheckBoxOn(name) Var<bool>(name, true, true)
#define CheckBoxOff(name) Var<bool>(name, false, true)

using namespace std;
using namespace pangolin;

GlViewer::GlViewer(std::string title, int w, int h, Eigen::Matrix3f K) :
		windowTitle(title), map(NULL), bufferSizeImage(0),
		bufferSizeVertices(0), slam(NULL), bufferSizeTriangles(0)
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
			MapStruct::MaxVertices,
			GL_FLOAT, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Initialise vertex array for coloured normal
	bufferNormals.Reinitialise(
			GlArrayBuffer,
			MapStruct::MaxVertices,
			GL_FLOAT, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Initialise vertex array for shaded rgb
	bufferTexture.Reinitialise(
			GlArrayBuffer,
			MapStruct::MaxVertices,
			GL_UNSIGNED_BYTE, 3,
			cudaGraphicsMapFlagsWriteDiscard,
			GL_STREAM_DRAW);

	// Bind vertex array to CUDA
	meshVerticesCUDAMapped = new CudaScopedMappedPtr(bufferVertices);
	meshNormalsCUDAMapped = new CudaScopedMappedPtr(bufferNormals);
	meshTextureCUDAMapped = new CudaScopedMappedPtr(bufferTexture);

	// Initialise texture array
	imageRGB.Reinitialise(w, h, GL_R8, true, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
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

	Display("SubDisplay_map").SetBounds(0.0, 1.0,  Attach::Pix(200), 1.0).
			SetLayout(LayoutOverlay).
			AddDisplay(imageBirdsEyeView).
			AddDisplay(modelViewCamera);
	Display("SubDisplay_images").SetBounds(0.0, 1.0, 0.75, 1.0).
			SetLayout(LayoutEqualVertical).
			AddDisplay(imageSyntheticView).
			AddDisplay(imageDepthView).
			AddDisplay(imageRGBView);

	// create menu entry i.e. a bunch of buttons
	panelMainMenu = CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, Attach::Pix(200), true);
	buttonSystemReset = new PushButton("ui.System Reset");
	buttonShowPoseGraph = new CheckBoxOff("ui.Show Pose Graph");
	buttonShowKeyPoints = new CheckBoxOff("ui.Show Key Points");
	buttonRenderSceneMesh = new CheckBoxOn("ui.Toggle Mesh");
	buttonShowCurrentCamera = new CheckBoxOn("ui.Toggle Camera");
	buttonFollowCamera = new CheckBoxOff("ui.Fllow Camera");
	buttonRenderSceneNormal = new CheckBoxOff("ui.Show Normal");
	buttonRenderSceneRGB = new CheckBoxOff("ui.Show Color Map");
	buttonExportMeshToFile = new PushButton("ui.Export to File");
	buttonToggleWireFrame = new CheckBoxOff("ui.Toggle Wire Frame");
	buttonEnableRGBImage = new CheckBoxOn("ui.Color Image");
	buttonEnableDepthImage = new CheckBoxOn("ui.Depth Image");
	buttonEnableSyntheticView = new CheckBoxOn("ui.Rendered Image");
	buttonEnableGraphMatching = new CheckBoxOn("ui.Graph Matching");
	buttonToggleLocalisationMode = new CheckBoxOff("ui.Localisation Only");
	buttonEnableBirdsEyeView = new CheckBoxOff("ui.Top Down View");
	buttonWriteMapToDiskBinary = new PushButton("ui.Write Map to Disk");
	buttonReadMapFromDiskBinary = new PushButton("ui.Read Map From Disk");

	// Key Bindings
	// CTL + r / R to restart the system.
	RegisterKeyPressCallback(PANGO_CTRL + 'r', SetVarFunctor<bool>("ui.System Reset", true));
	RegisterKeyPressCallback(PANGO_CTRL + 'R', SetVarFunctor<bool>("ui.System Reset", true));

	// unbind current context from the main thread
	pangolin::GetBoundWindow()->RemoveCurrent();
}

GlViewer::~GlViewer()
{
	pangolin::GetBoundWindow()->RemoveCurrent();

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
	delete buttonEnableGraphMatching;
	delete buttonToggleLocalisationMode;
	delete buttonEnableBirdsEyeView;
	delete buttonWriteMapToDiskBinary;
	delete buttonReadMapFromDiskBinary;

	pangolin::DestroyWindow(windowTitle);
}

void GlViewer::drawKeyPointsToScreen() const
{

}

void GlViewer::setCurrentImages(PointCloud* data)
{
	SafeCall(cudaMemcpy2DToArray(**imageRGBCUDAMapped, 0, 0,
		(void* )data->image[0].data, data->image[0].step,
		sizeof(uchar) * data->image[0].cols, data->image[0].rows,
		cudaMemcpyDeviceToDevice));

	SafeCall(cudaMemcpy2DToArray(**imageDepthCUDAMapped, 0, 0,
		(void* )data->depth[0].data, data->depth[0].step,
		sizeof(float) * data->image[0].cols, data->image[0].rows,
		cudaMemcpyDeviceToDevice));
}

void GlViewer::processMessages()
{
	// if SYSTEM RESET
	if (Pushed(*buttonSystemReset))
	{
		*buttonToggleLocalisationMode = false;
		slam->queueMessage(Msg(Msg::SYSTEM_RESET));
	}

	// if EXPORT MESH TO FILE
	if (Pushed(*buttonExportMeshToFile))
		slam->queueMessage(Msg(Msg::EXPORT_MESH_TO_FILE));

	// if WRITE MAP TO BINARY FILE
	if (Pushed(*buttonWriteMapToDiskBinary))
	{
		slam->queueMessage(Msg(Msg::WRITE_BINARY_MAP_TO_DISK));
	}

	// if READ MAP FROM BINARY FILE
	if (Pushed(*buttonReadMapFromDiskBinary))
	{
		slam->queueMessage(Msg(Msg::READ_BINARY_MAP_FROM_DISK));
		*buttonToggleLocalisationMode = true;
	}
}

void GlViewer::drawViewsToScreen()
{
	if(pangolin::ShouldQuit())
	{
		return;
	}

	glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	modelViewCamera.Activate(viewCam);

	if (*buttonShowPoseGraph)
	{
		drawKeyFrameGraph();
	}

	if (*buttonShowKeyPoints)
	{
		drawKeyPointsToScreen();
	}

	if (*buttonToggleWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}

	if (*buttonRenderSceneMesh)
	{
		if (*buttonRenderSceneNormal)
			*buttonRenderSceneNormal = false;
		if (*buttonRenderSceneRGB)
			*buttonRenderSceneRGB = false;
		drawShadedMesh(false);
	}

	if (*buttonRenderSceneNormal)
	{
		if (*buttonRenderSceneRGB)
			*buttonRenderSceneRGB = false;
		drawShadedMesh(true);
	}

	if (*buttonRenderSceneRGB)
	{
		drawTexturedMesh();
	}

	if (*buttonToggleWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	if (*buttonShowCurrentCamera)
	{
		drawCurrentCamera();
	}

	if (buttonFollowCamera->Get())
	{
		setModelViewFollowCamera();
	}

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

void GlViewer::drawCurrentCamera() const
{

	std::vector<GLfloat> cam = getTransformedCam(currentCamPose);
	glColor3fv(RGBActiveCam);
	glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_LINE_STRIP, 3);
}

void GlViewer::drawKeyFrameGraph() const
{
	std::vector<GLfloat> node;
	for (SE3 pose : keyFrameGraph)
	{
		std::vector<GLfloat> cam = getTransformedCam(pose, 0.5);
		node.push_back(pose.translation()(0));
		node.push_back(pose.translation()(1));
		node.push_back(pose.translation()(2));
		glColor3fv(RGBKeyFrameGraph);
		glDrawVertices(cam.size() / 3, (GLfloat*) &cam[0], GL_LINE_STRIP, 3);
	}

	glColor3f(0.0f, 1.0f, 0.0f);
	glDrawVertices(node.size() / 3, (GLfloat*) &node[0], GL_LINE_STRIP, 3);
}

std::vector<GLfloat> GlViewer::getTransformedCam(SE3 pose, float scale) const
{
	std::vector<GLfloat> result;

	// Generate a transformed array of points
	// which represents the camera wire-frame
	for (Eigen::Vector3f vertex : camVertices)
	{
		Eigen::Vector3f vertex_transformed = pose.rotationMatrix().cast<float>() * vertex * scale + pose.translation().cast<float>();
		result.push_back(vertex_transformed(0));
		result.push_back(vertex_transformed(1));
		result.push_back(vertex_transformed(2));
	}

	return result;
}

void GlViewer::drawBirdsEyeViewToCamera() const
{
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageBirdsEye.RenderToViewport(true);
}

void GlViewer::drawSyntheticViewToCamera() const
{
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageSynthetic.RenderToViewport(true);
}

void GlViewer::drawDepthViewToCamera() const
{

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageDepth.RenderToViewport(true);
}

void GlViewer::drawRGBViewToCamera() const
{
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	imageRGB.RenderToViewport(true);
}

void GlViewer::drawTexturedMesh()
{
	if (map->meshUpdated)
	{
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

void GlViewer::drawShadedMesh(bool showNormals)
{

	if (map->noTrianglesHost == 0)
		return;

	if (map->meshUpdated) {
		cudaMemcpy((void*) **meshVerticesCUDAMapped, (void*) map->modelVertex, sizeof(float3) * map->noTrianglesHost * 3,  cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshNormalsCUDAMapped, (void*) map->modelNormal, sizeof(float3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy((void*) **meshTextureCUDAMapped, (void*) map->modelColor, sizeof(uchar3) * map->noTrianglesHost * 3, cudaMemcpyDeviceToDevice);
		map->meshUpdated = false;
	}

	pangolin::GlSlProgram * program;
	if (showNormals)
		program = &shaderNormal;
	else
		program = &shaderPhong;

	program->SaveBind();
	program->SetUniform("viewMat", viewCam.GetModelViewMatrix());
	program->SetUniform("projMat", viewCam.GetProjectionMatrix());
	Eigen::Vector3f translation = currentCamPose.translation().cast<float>();
	program->SetUniform("lightpos", translation(0), translation(1), translation(2));

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

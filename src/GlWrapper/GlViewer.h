#pragma once

#include "SlamSystem.h"
#include "Utilities/SophusUtil.h"
#include <atomic>
#include <vector>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

class SlamSystem;
class Tracker;
class DenseMap;

class GlViewer
{
public:

	void spin();
	void run() {}

	void setMap(DenseMap * pMap);
	void setSystem(SlamSystem * pSystem);

	void drawKeys();
	void drawNormal();
	void drawColor();
	void drawMesh(bool bNormal);
	void drawRGBViewToCamera();
	void drawSyntheticViewToCamera();
	void drawDepthViewToCamera();
	void drawBirdsEyeViewToCamera();

	DenseMap* map;
	SlamSystem* system;

	pangolin::OpenGlMatrix pose;

	//============= REFACTORING =============

	GlViewer(std::string windowTitle, int w, int h, Eigen::Matrix3f K);
	GlViewer(const GlViewer&) = delete;
	GlViewer& operator=(const GlViewer&) = delete;
	~GlViewer();

	void processMessages();
	void drawViewsToScreen();
	void setCurrentImages(PointCloud* data);

	inline void destoryWindow() const;
	inline bool shouldQuit() const;
	inline void enableGlContext() const;
	inline void disableGlContext() const;
	inline void setCurrentCamPose(SE3 pose);
	inline void setKeyFrameGraph(std::vector<SE3>&);

	std::mutex mutexMeshUpdate;

	// OpenGL array mapped to CUDA
	pangolin::CudaScopedMappedPtr* meshVerticesCUDAMapped;
	pangolin::CudaScopedMappedPtr* meshNormalsCUDAMapped;
	pangolin::CudaScopedMappedPtr* meshTextureCUDAMapped;

	// OpenGL textures mapped to CUDA
	pangolin::CudaScopedMappedArray* imageRGBCUDAMapped;
	pangolin::CudaScopedMappedArray* imageDepthCUDAMapped;
	pangolin::CudaScopedMappedArray* imageSyntheticCUDAMapped;
	pangolin::CudaScopedMappedArray* imageBirdsEyeCUDAMapped;

protected:

	// utility functions
	void drawCurrentCamera() const;
	void drawKeyFrameGraph() const;
	void setModelViewFollowCamera();
	std::vector<GLfloat> getTransformedCam(SE3 pose) const;

	pangolin::View modelViewCamera;
	pangolin::View imageSyntheticView;
	pangolin::View imageDepthView;
	pangolin::View imageRGBView;
	pangolin::View imageBirdsEyeView;

	// a series of const variables
	const std::string windowTitle;
	const float RGBKeyFrameGraph[3] = { 0.0f, 0.0f, 1.0f };
	const float RGBActiveCam[3] = { 0.0f, 1.0f, 0.0f };
	const float RGBInactiveCam[3] = { 1.0f, 0.0f, 0.0f };

	// camera vertices array
	const Eigen::Vector3f camVertices[12] =
	{
		{ 0.04,  0.03,  0    },
		{ 0.04, -0.03,  0    },
		{ 0,     0,    -0.03 },
		{-0.04,  0.03,  0    },
		{-0.04, -0.03,  0    },
		{ 0,     0,    -0.03 },
		{ 0.04,  0.03,  0    },
		{-0.04,  0.03,  0    },
		{ 0, 	 0,    -0.03 },
		{ 0.04, -0.03,  0    },
		{-0.04, -0.03,  0    },
		{ 0,     0,    -0.03 }
	};

	// menus and buttons
	pangolin::View panelMainMenu;
	pangolin::Var<bool>* buttonSystemReset;
	pangolin::Var<bool>* buttonShowPoseGraph;
	pangolin::Var<bool>* buttonShowKeyPoints;
	pangolin::Var<bool>* buttonRenderSceneMesh;
	pangolin::Var<bool>* buttonShowCurrentCamera;
	pangolin::Var<bool>* buttonFollowCamera;
	pangolin::Var<bool>* buttonRenderSceneNormal;
	pangolin::Var<bool>* buttonRenderSceneRGB;
	pangolin::Var<bool>* buttonExportMeshToFile;
	pangolin::Var<bool>* buttonToggleWireFrame;
	pangolin::Var<bool>* buttonEnableRGBImage;
	pangolin::Var<bool>* buttonEnableDepthImage;
	pangolin::Var<bool>* buttonEnableSyntheticView;
	pangolin::Var<bool>* buttonPauseSystem;
	pangolin::Var<bool>* buttonEnableGraphMatching;
	pangolin::Var<bool>* buttonToggleLocalisationMode;
	pangolin::Var<bool>* buttonEnableBirdsEyeView;
	pangolin::Var<bool>* buttonWriteMapToDiskBinary;
	pangolin::Var<bool>* buttonReadMapFromDiskBinary;

	// Vertex array object for rendering
	GLuint vaoFULL;
	GLuint vaoVerticesAndNormal;

	// Shading programs;
	pangolin::GlSlProgram shaderPhong;
	pangolin::GlSlProgram shaderNormal;
	pangolin::GlSlProgram shaderTexture;

	// OpenGL and CUDA inter-operate buffers
	pangolin::GlBufferCudaPtr bufferVertices;
	pangolin::GlBufferCudaPtr bufferNormals;
	pangolin::GlBufferCudaPtr bufferTexture;

	// OpenGL texture used for visualisation
	pangolin::GlTextureCudaArray imageRGB;
	pangolin::GlTextureCudaArray imageDepth;
	pangolin::GlTextureCudaArray imageSynthetic;
	pangolin::GlTextureCudaArray imageBirdsEye;

	pangolin::OpenGlRenderState viewCam;

	// Current camera pose
	SE3 currentCamPose;

	// KeyFrame Poses
	std::vector<SE3> keyFrameGraph;

	int bufferSizeTriangles;
	int bufferSizeVertices;
	int bufferSizeImage;
};

inline void GlViewer::disableGlContext() const
{
	pangolin::GetBoundWindow()->RemoveCurrent();
}

inline void GlViewer::destoryWindow() const
{
	pangolin::DestroyWindow(windowTitle);
}

inline bool GlViewer::shouldQuit() const
{
	return pangolin::ShouldQuit();
}

inline void GlViewer::enableGlContext() const
{
	pangolin::BindToContext(windowTitle);
}

inline void GlViewer::setCurrentCamPose(SE3 pose)
{
	currentCamPose = pose;
}

inline void GlViewer::setKeyFrameGraph(std::vector<SE3>& graph)
{
	keyFrameGraph = graph;
}

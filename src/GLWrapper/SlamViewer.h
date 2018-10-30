#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include "System.h"

#include <atomic>
#include <vector>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include "Utilities/SophusUtil.h"

class System;
class Tracker;
class DistanceField;

class SlamViewer
{
public:

	SlamViewer();
	void spin();
	void run() {}

	void setMap(DistanceField * pMap);
	void setSystem(System * pSystem);
	void setTracker(Tracker * pTracker);
	void signalQuit();
	void Insert(std::vector<GLfloat> & vPt, Eigen::Vector3f & pt);
	void drawCamera();
	void drawKeys();
	void followCam();
	void drawNormal();
	void drawColor();
	void drawKeyFrame();
	void drawMesh(bool bNormal);
	void showColorImage();
	void showPrediction();
	void showDepthImage();
	void topDownView();

	System * system;
	DistanceField * map;
	Tracker * tracker;

	std::atomic<bool> quit;


	pangolin::OpenGlMatrix pose;

	//============= REFACTORING =============

	SlamViewer(int w, int h, Eigen::Matrix3f K, std::string title);

	inline void setCurrentCamPose(SE3 pose);
	inline void setKeyFramePoses(std::vector<SE3> & poses);

	// OpenGL array mapped to CUDA
	std::mutex mutexMeshUpdate;
	pangolin::CudaScopedMappedPtr* meshVerticesCUDAMapped;
	pangolin::CudaScopedMappedPtr* meshNormalsCUDAMapped;
	pangolin::CudaScopedMappedPtr* meshTextureCUDAMapped;

	// OpenGL textures mapped to CUDA
	pangolin::CudaScopedMappedArray* imageTextureCUDAMapped;
	pangolin::CudaScopedMappedArray* imageDepthCUDAMapped;
	pangolin::CudaScopedMappedArray* imageSyntheticViewCUDAMapped;
	pangolin::CudaScopedMappedArray* imageTopdownViewCUDAMapped;

protected:

	std::vector<GLfloat> getTransformedCamera(SE3 pose) const;

	// Vertex array object for rendering
	GLuint vertexArrayObjectMesh;
	GLuint vertexArrayObjectColor;

	// Shading programs;
	pangolin::GlSlProgram shaderPhong;
	pangolin::GlSlProgram shaderNormalmap;
	pangolin::GlSlProgram shaderTexture;

	// OpenGL and CUDA inter-operate buffers
	pangolin::GlBufferCudaPtr meshVertices;
	pangolin::GlBufferCudaPtr meshNormals;
	pangolin::GlBufferCudaPtr meshTexture;

	// Keep a copy of buffers to prevent flickering
	pangolin::GlBufferCudaPtr meshVerticesOld;
	pangolin::GlBufferCudaPtr meshNormalsOld;
	pangolin::GlBufferCudaPtr meshTextureOld;

	// OpenGL texture used for visualisation
	pangolin::GlTextureCudaArray imageTexture;
	pangolin::GlTextureCudaArray imageDepth;
	pangolin::GlTextureCudaArray imageSyntheticView;
	pangolin::GlTextureCudaArray imageTopdownView;

	pangolin::OpenGlRenderState viewCam;

	// Current camera position
	SE3 currentCamPose;

	// KeyFrame Poses used for drawing
	std::vector<SE3> keyFramePoses;

	int bufferSizeTriangles;
	int bufferSizeVertices;
	int bufferSizeImage;

	const Eigen::Vector3f camVertices[12] =
	{
		{ 0.04, 0.03, 0    },
		{ 0.04,-0.03, 0    },
		{ 0,    0,   -0.03 },
		{-0.04, 0.03, 0    },
		{-0.04,-0.03, 0    },
		{ 0,    0,   -0.03 },
		{ 0.04, 0.03, 0    },
		{-0.04, 0.03, 0    },
		{ 0, 	0, 	 -0.03 },
		{ 0.04,-0.03, 0    },
		{-0.04,-0.03, 0    },
		{ 0,    0,   -0.03 }
	};
};

inline void SlamViewer::setCurrentCamPose(SE3 pose)
{
	currentCamPose = pose;
}

inline void SlamViewer::setKeyFramePoses(std::vector<SE3> & poses)
{
	keyFramePoses = poses;
}

#endif

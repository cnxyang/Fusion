#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include <vector>
#include "Mapping.hpp"
#include "System.hpp"
#include "Tracking.hpp"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

class System;
class Mapping;
class tracker;

class Viewer {
public:

	Viewer();
	void spin();

	void setMap(Mapping* pMap);
	void setSystem(System* pSystem);
	void setTracker(tracker* pTracker);

private:

	void Insert(std::vector<GLfloat>& vPt, Eigen::Vector3f& pt);
	void drawCamera();
	void drawKeys();
	void followCam();
	void drawNormal();
	void drawColor();
	void drawTrajectory();
	void drawMesh(bool bNormal);

	Mapping* mpMap;

	GLuint vao;
	System* psystem;
	tracker* ptracker;
	pangolin::OpenGlRenderState sCam;
	pangolin::GlSlProgram phongShader;
	pangolin::GlSlProgram normalShader;
	pangolin::GlSlProgram colorShader;
	pangolin::GlBufferCudaPtr vertex;
	pangolin::GlBufferCudaPtr normal;
	pangolin::GlBufferCudaPtr color;
	pangolin::CudaScopedMappedPtr * vertexMaped;
	pangolin::CudaScopedMappedPtr * normalMaped;
	pangolin::CudaScopedMappedPtr * colorMaped;
};

#endif

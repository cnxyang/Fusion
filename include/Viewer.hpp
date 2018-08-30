#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include <vector>
#include "Mapping.hpp"
#include "System.hpp"
#include "Tracking.hpp"
#include <pangolin/pangolin.h>

class System;
class Tracking;

class Viewer {
public:

	Viewer();
	void Spin();

	void SetMap(Mapping* pMap);
	void SetSystem(System* pSystem);
	void SetTracker(Tracking* pTracker);

private:

	void Insert(std::vector<GLfloat>& vPt, Eigen::Vector3d& pt);
	void DrawCamera();
	void DrawKeys();
	void FollowCam();
	void DrawNormal();
	void DrawColor();
	void DrawTrajectory();
	void DrawMesh(bool bNormal);

	Mapping* mpMap;
	System* mpSystem;
	Tracking* mpTracker;
	GLuint vao;
	Eigen::Matrix4d T;
	pangolin::OpenGlRenderState s_cam;
	pangolin::GlSlProgram mPhongProg;
	pangolin::GlSlProgram mNormalProg;
	pangolin::GlSlProgram mColorProg;
	pangolin::GlBufferCudaPtr array;
	pangolin::GlBufferCudaPtr normal;
	pangolin::GlBufferCudaPtr color;
	pangolin::CudaScopedMappedPtr* var;
	pangolin::CudaScopedMappedPtr* nvar;
	pangolin::CudaScopedMappedPtr* cvar;
	bool mbShowMesh;
	int n = 0;
};

#endif

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
	void DrawMesh();
	void DrawKeys();
	void DrawTrajectory();

	Mapping* mpMap;
	System* mpSystem;
	Tracking* mpTracker;
};

#endif

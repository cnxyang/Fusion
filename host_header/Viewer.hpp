#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include "Mapping.hpp"
#include "System.hpp"
#include "Tracking.hpp"

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

	void DrawTrajectory();

	Mapping* mpMap;
	System* mpSystem;
	Tracking* mpTracker;
};

#endif

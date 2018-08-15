#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include "Mapping.hpp"
#include "System.hpp"
#include "Tracking.hpp"

class System;

class Viewer {
public:

	Viewer();
	void Spin();

	void SetMap(Mapping* pMap);
	void SetSystem(System* pSystem);
	void SetTracker(Tracking* pTracker);

private:

	Mapping* mpMap;
	System* mpSystem;
	Tracking* mpTracker;
};

#endif

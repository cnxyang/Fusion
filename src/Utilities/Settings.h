#pragma once


struct GlobalStates
{
	GlobalStates();

	// Used for tracking
	int numTrackedFrames;
	int numTrackedKeyFrames;
};

extern GlobalStates systemState;
extern bool displayDebugInfo;

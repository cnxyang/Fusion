#include "Settings.h"

SystemState::SystemState()
{
	numTrackedFrames = 0;
	numTrackedKeyFrames = 0;
	showGeneratedMesh = true;
	showInputImages = true;
	localisatonOnly = false;
}

SystemState systemState;

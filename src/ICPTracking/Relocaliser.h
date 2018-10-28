#pragma once

#include <Eigen/Core>

class ICPTracker;

class Relocaliser
{
public:

	Relocaliser(int w, int h, Eigen::Matrix3f K);

private:

	int maxNumAttempts;
	ICPTracker* relocaliseTracker;
};

#ifndef MAP_POINT__
#define MAP_POINT__

#include <map>
#include <Eigen/Dense>

#include "KeyFrame.h"

class KeyFrame;

struct MapPoint {

	MapPoint(const KeyFrame * kf);

	Eigen::Vector3f GetWorldPosition();

	Eigen::Vector3f position;
	std::map<const KeyFrame *, int> observations;

	const KeyFrame * RefKF;

	bool bad;
	int pointId;
	static int nextId;
};

#endif

#ifndef MAP_POINT__
#define MAP_POINT__

#include <map>
#include <Eigen/Dense>

#include "KeyFrame.h"

class KeyFrame;

struct MapPoint {

	MapPoint();

	Eigen::Vector3f position;
	std::map<const KeyFrame *, int> observations;

	bool bad;
	int pointId;
	static int nextId;
};

#endif

#include "MapPoint.h"

int MapPoint::nextId = 0;

MapPoint::MapPoint(const KeyFrame * kf):
	bad(false) {
	pointId = nextId++;
	RefKF = kf;
}

Eigen::Vector3f MapPoint::GetWorldPosition() {
//	return RefKF->Rotation() * position + RefKF->Translation();
}

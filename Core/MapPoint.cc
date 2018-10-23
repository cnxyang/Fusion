#include "MapPoint.h"

int MapPoint::nextId = 0;

MapPoint::MapPoint():
	bad(false) {
	pointId = nextId++;
}

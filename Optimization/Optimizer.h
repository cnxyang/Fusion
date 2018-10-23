#ifndef OPTIMIZER_H__
#define OPTIMIZER_H__

#include "System.h"
#include "Mapping.h"

class System;

class Optimizer {

public:

	const int NUM_LOCAL_KF = 7;

	Optimizer();

	void run();

	void LocalBA();

	void GlobalBA();

	void GetLocalMap();

	static int OptimizePose(Frame * f, std::vector<Eigen::Vector3d> & points,
			std::vector<Eigen::Vector2d> & obs, Eigen::Matrix4d & dt);

	void SetMap(Mapping * map_);

	void SetSystem(System * sys_);

protected:

	bool CheckLoop();

	void CloseLoop();

	Mapping * map;

	System * sys;

	size_t noKeyFrames;

	KeyFrame * currentKF;

	std::vector<KeyFrame *> localMap;

	std::vector<KeyFrame *> globalMap;

	std::vector<MapPoint *> mapPoints;
};

#endif

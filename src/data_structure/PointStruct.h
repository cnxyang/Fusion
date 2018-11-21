#pragma once

class PointStruct
{
	Eigen::Vector3f point;
	std::map<Frame*, int> observations;
};

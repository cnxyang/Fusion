#pragma once
#include <Eigen/Core>
#include "VectorMath.h"

typedef float NumType;
typedef Eigen::Matrix<NumType, 6, 6> Matrix6x6;

inline Matrix3f EigenToMatrix3f(Eigen::Matrix3d input)
{
	Matrix3f mat3f;
	mat3f.rowx = make_float3((float) input(0, 0), (float) input(0, 1), (float) input(0, 2));
	mat3f.rowy = make_float3((float) input(1, 0), (float) input(1, 1), (float) input(1, 2));
	mat3f.rowz = make_float3((float) input(2, 0), (float) input(2, 1), (float) input(2, 2));
	return mat3f;
}

inline float3 EigenToFloat3(Eigen::Vector3d input)
{
	return make_float3((float) input(0), (float) input(1), (float) input(2));
}

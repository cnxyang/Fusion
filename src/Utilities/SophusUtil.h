#pragma once
#include <sophus/se3.hpp>
#include "Utilities/VectorMath.h"

typedef Sophus::SE3d SE3;

inline Matrix3f SE3toMatrix3f(SE3 in)
{
	Matrix3f mat3f;
	Eigen::Matrix3d rotationMatrix = in.rotationMatrix();
	mat3f.rowx = make_float3(rotationMatrix(0, 0), rotationMatrix(0, 1), rotationMatrix(0, 2));
	mat3f.rowy = make_float3(rotationMatrix(1, 0), rotationMatrix(1, 1), rotationMatrix(1, 2));
	mat3f.rowz = make_float3(rotationMatrix(2, 0), rotationMatrix(2, 1), rotationMatrix(2, 2));
	return mat3f;
}

inline float3 SE3toFloat3(SE3 in)
{
	return make_float3(in.translation()(0), in.translation()(1), in.translation()(2));
}

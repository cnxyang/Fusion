#ifndef CONVERTER_H__
#define CONVERTER_H__

static Matrix3f eigen_to_mat3f(Eigen::Matrix3d mat) {
	Matrix3f mat3f;
	mat3f.rowx = { (float) mat(0, 0), (float) mat(0, 1), (float) mat(0, 2)};
	mat3f.rowx = { (float) mat(1, 0), (float) mat(1, 1), (float) mat(1, 2)};
	mat3f.rowx = { (float) mat(2, 0), (float) mat(2, 1), (float) mat(2, 2)};
	return mat3f;
}

static float3 eigen_to_float3(Eigen::Vector3d vec) {
	float3 vec3f = { (float) vec(0), (float) vec(1), (float) vec(2) };
	return vec3f;
}

#endif

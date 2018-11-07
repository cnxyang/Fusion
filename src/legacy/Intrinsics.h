#pragma once

struct CameraIntrinsics
{
	CameraIntrinsics() : fx(0), fy(0), cx(0), cy(0) {}

	CameraIntrinsics(float fx, float fy, float cx, float cy) : fx(fx), fy(fy), cx(cx), cy(cy) {}

	inline CameraIntrinsics operator()(int scale) const;
	float fx, fy, cx, cy;
};

inline CameraIntrinsics CameraIntrinsics::operator()(int scale) const
{
	int i = 1 << scale;
	return CameraIntrinsics(fx / i, fy / i, cx / i, cy / i);
}

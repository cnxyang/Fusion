/*
 * VecMath.h
 *
 *  Created on: 13 Apr 2018
 *      Author: xy
 */

#ifndef VECMATH_H_
#define VECMATH_H_

#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

/* ------------------------------------ *
 * forward declaration(host and device) *
 *------------------------------------- */

__host__  __device__  inline float3 make_float3(float);

/* ----------------------------- *
 * forward declaration( device ) *
 *------------------------------ */

#ifdef __CUDACC__

__device__  inline int3 make_int3(float3);

#endif

/* ---------------- *
 * make_uchar(num)  *
 *----------------- */

#ifdef __CUDACC__

__device__  inline uchar3 make_uchar3(float3 a) {
	return make_uchar3(__float2int_rd(a.x), __float2int_rd(a.y),
			__float2int_rd(a.z));
}

#endif

/* -------------- *
 * make_int(num)  *
 *--------------- */

__host__  __device__  inline int2 make_int2(int a) {
	return make_int2(a, a);
}

__device__  inline int2 make_int2(float2 a) {
#ifdef __CUDACC__
	return make_int2(__float2int_rd(a.x), __float2int_rd(a.y));
#else
	return make_int2(static_cast<int>(a.x),
			static_cast<int>(a.y));
#endif
}

__host__  __device__  inline int3 make_int3(int a) {
	return make_int3(a, a, a);
}

#ifdef __CUDACC__

__device__  inline int3 make_int3(float a) {
	return make_int3(make_float3(a));
}

__device__  inline int3 make_int3(float3 a) {
	return make_int3(__float2int_rd(a.x), __float2int_rd(a.y),
			__float2int_rd(a.z));
}

#endif

/* -------------- *
 * make_uint(num) *
 *--------------- */

__host__  __device__  inline uint2 make_uint2(int a) {
	return make_uint2(a, a);
}

__host__  __device__  inline uint3 make_uint3(int a) {
	return make_uint3(a, a, a);
}

/* ---------------- *
 * make_float(num)  *
 *----------------- */

__host__  __device__  inline float2 make_float2(float a) {
	return make_float2(a, a);
}

__host__  __device__  inline float3 make_float3(uchar3 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__  __device__  inline float3 make_float3(float a) {
	return make_float3(a, a, a);
}

__host__  __device__  inline float3 make_float3(int3 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__  __device__  inline float3 make_float3(float4 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__  __device__  inline float4 make_float4(float a) {
	return make_float4(a, a, a, a);
}

__host__  __device__  inline float4 make_float4(float3 a) {
	return make_float4(a.x, a.y, a.z, 1.f);
}

__host__  __device__  inline float4 make_float4(float3 a, float b) {
	return make_float4(a.x, a.y, a.z, b);
}

/* ---------------- *
 * make_double(num) *
 *----------------- */

__host__  __device__  inline double4 make_double4(double a) {
	return make_double4(a, a, a, a);
}

/* ----------------- *
 * operator+(type)   *
 *------------------ */

__host__  __device__  inline int2 operator+(int2 a, int2 b) {
	return make_int2(a.x + b.x, a.y + b.y);
}

__host__  __device__  inline float2 operator+(float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__host__  __device__  inline uchar3 operator+(uchar3 a, uchar3 b) {
	return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__  __device__  inline int3 operator+(int3 a, int3 b) {
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__  __device__  inline float3 operator+(float3 a, float b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

__host__  __device__  inline float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__  __device__  inline float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/* ----------------- *
 * operator+=(type)  *
 *------------------ */

__host__ __device__ inline
void operator+=(float3 & a, uchar3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__ inline
void operator+=(float3 & a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__ inline
void operator-=(float3 & a, float3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

/* ----------------- *
 * operator-(type)   *
 *------------------ */

__host__  __device__  inline int2 operator-(int2 a, int2 b) {
	return make_int2(a.x - b.x, a.y - b.y);
}

__host__  __device__  inline float3 operator-(float3 b) {
	return make_float3(-b.x, -b.y, -b.z);
}

__host__  __device__  inline float3 operator-(float3 a, float b) {
	return make_float3(a.x - b, a.y - b, a.z - b);
}

__host__  __device__  inline float3 operator-(float a, float3 b) {
	return make_float3(a - b.x, a - b.y, a - b.z);
}

__host__  __device__  inline float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__  __device__  inline float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/* ------------------- *
 * operator(vec)*(vec) *
 *-------------------- */

__host__ __device__ inline
float operator*(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
float operator*(float3 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
float operator*(float4 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w;
}

__host__ __device__ inline
float operator*(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* ---------------------- *
 * operator(vec)*(scalar) *
 *----------------------- */

__host__  __device__  inline uchar3 operator*(uchar3 a, unsigned short b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline uchar3 operator*(uchar3 a, int b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline uchar3 operator*(int b, uchar3 a) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline int3 operator*(int3 a, uint b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline int3 operator*(int3 a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline float3 operator*(int3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__  __device__  inline float3 operator*(float a, float3 b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__  __device__  inline float4 operator*(float4 a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/* ------------------- *
 * operator(vec)/(vec) *
 *-------------------- */

__host__  __device__  inline int3 operator/(int3 a, int3 b) {
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__  __device__  inline float3 operator/(float3 a, int3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__  __device__  inline float3 operator/(float3 a, float3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__  __device__  inline float4 operator/(float4 a, float4 b) {
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

/* ---------------------- *
 * operator(vec)/(scalar) *
 *----------------------- */

__host__  __device__  inline int2 operator/(int2 a, int b) {
	return make_int2(a.x / b, a.y / b);
}

__host__  __device__  inline uchar3 operator/(uchar3 a, int b) {
	return make_uchar3(a.x / b, a.y / b, a.z / b);
}

__host__  __device__  inline int3 operator/(int3 a, uint b) {
	return make_int3(a.x / (int) b, a.y / (int) b, a.z / (int) b);
}

__host__  __device__  inline int3 operator/(int3 a, int b) {
	return make_int3(a.x / b, a.y / b, a.z / b);
}

__host__  __device__  inline float3 operator/(float3 a, int b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__  __device__  inline float3 operator/(float3 a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__  __device__  inline float3 operator/(float a, float3 b) {
	return make_float3(a / b.x, a / b.y, a / b.z);
}

__host__  __device__  inline float4 operator/(float4 a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

/* ---------------------- *
 * operator(vec)%(scalar) *
 *----------------------- */

__host__  __device__  inline int3 operator%(int3 a, int b) {
	return make_int3(a.x % b, a.y % b, a.z % b);
}

/* ---------------------- *
 * operator(vec) == (vec) *
 *----------------------- */

__host__ __device__ inline
bool operator==(int3 a, int3 b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

#ifdef __CUDACC__
/* ---------------- *
 * cross(type(num)) *
 *----------------- */

__device__  inline float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

__device__  inline float3 cross(float4 a, float4 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

/* ---------------- *
 * max(float(num))  *
 *----------------- */

__device__ static inline
void atomicMax(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

/* ---------------- *
 * min(float(num))  *
 *----------------- */

__device__ static inline
void atomicMin(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

/* ------------ *
 *  norm(type)  *
 *------------- */

__device__ inline
float norm(float3 a) {
	return sqrtf(a * a);
}

__device__ inline
float norm(float4 a) {
	return sqrtf(a * a);
}

/* -------------------- *
 *  inverse norm(type)  *
 *--------------------- */

__device__ inline
float inv_norm(float3 a) {
	return rsqrtf(a * a);
}

__device__ inline
float inv_norm(float4 a) {
	return rsqrtf(a * a);
}

/* ----------------- *
 *  Normalised(type) *
 *------------------ */

__device__  inline float3 normalised(float3 a) {
	return a / norm(a);
}

__device__  inline float4 normalised(float4 a) {
	return a / norm(a);
}

/*------------- *
 *  floor(vec)  *
 *------------- */

__device__  inline float3 floor(float3 a) {
	return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

/*------------- *
 *  interp(num) *
 *------------- */

#endif

struct Matrix3f {
	__host__ __device__ Matrix3f() {
		rowx = rowy = rowz = make_float3(0, 0, 0);
	}

	__host__
	Matrix3f(cv::Mat& mat) {
		rowx = make_float3(mat.at<float>(0, 0), mat.at<float>(0, 1), mat.at<float>(0, 2));
		rowy = make_float3(mat.at<float>(1, 0), mat.at<float>(1, 1), mat.at<float>(1, 2));
		rowz = make_float3(mat.at<float>(2, 0), mat.at<float>(2, 1), mat.at<float>(2, 2));
	}

	__host__
	Matrix3f(const cv::Mat& mat) {
		rowx = make_float3(mat.at<float>(0, 0), mat.at<float>(0, 1), mat.at<float>(0, 2));
		rowy = make_float3(mat.at<float>(1, 0), mat.at<float>(1, 1), mat.at<float>(1, 2));
		rowz = make_float3(mat.at<float>(2, 0), mat.at<float>(2, 1), mat.at<float>(2, 2));
	}

	__host__ __device__ inline
	static Matrix3f Identity() {
		Matrix3f id;
		id.rowx = make_float3(1, 0, 0);
		id.rowy = make_float3(0, 1, 0);
		id.rowz = make_float3(0, 0, 1);
		return id;
	}

	__host__ __device__  inline float3 operator*(float3 vec) {
		return make_float3(rowx * vec, rowy * vec, rowz * vec);
	}

	__host__ __device__  inline float3 operator*(float3 vec) const {
		return make_float3(rowx * vec, rowy * vec, rowz * vec);
	}

	__host__ __device__  inline float4 operator*(float4 vec) {
		return make_float4(rowx * vec, rowy * vec, rowz * vec, vec.w);
	}

	__host__ __device__  inline float4 operator*(float4 vec) const {
		return make_float4(rowx * vec, rowy * vec, rowz * vec, vec.w);
	}

	__host__ __device__ inline float3 coloumx() const {
		return make_float3(rowx.x, rowy.x, rowz.x);
	}

	__host__ __device__ inline float3 coloumy() const {
		return make_float3(rowx.y, rowy.y, rowz.y);
	}

	__host__ __device__ inline float3 coloumz() const {
		return make_float3(rowx.z, rowy.z, rowz.z);
	}

	float3 rowx;
	float3 rowy;
	float3 rowz;
};

#endif /* VECMATH_H_ */

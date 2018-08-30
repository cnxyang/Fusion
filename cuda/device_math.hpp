/*
 * VecMath.h
 *
 *  Created on: 13 Apr 2018
 *      Author: xy
 */

#ifndef DEVICE_MATH_HPP__
#define DEVICE_MATH_HPP__

#include <cmath>
#include <opencv.hpp>
#include <cuda_runtime.h>

#define HOST_FUNC __host__
#define DEV_FUNC __device__

/* ---------------- *
 * make_uchar(num)  *
 *----------------- */
DEV_FUNC uchar3 make_uchar3(float3 a);

/* -------------- *
 * make_int(num)  *
 *--------------- */
DEV_FUNC int2 make_int2(float2 a);
DEV_FUNC int3 make_int3(float a);
DEV_FUNC int3 make_int3(float3 a);
HOST_FUNC DEV_FUNC int2 make_int2(int a);
HOST_FUNC DEV_FUNC int3 make_int3(int a);
HOST_FUNC DEV_FUNC int4 make_int4(int3 a, int b);

/* -------------- *
 * make_uint(num) *
 *--------------- */
HOST_FUNC DEV_FUNC uint2 make_uint2(int a);
HOST_FUNC DEV_FUNC uint3 make_uint3(int a);

/* ---------------- *
 * make_float(num)  *
 *----------------- */
HOST_FUNC DEV_FUNC float2 make_float2(float a);
HOST_FUNC DEV_FUNC float3 make_float3(uchar3 a);
HOST_FUNC DEV_FUNC float3 make_float3(float a);
HOST_FUNC DEV_FUNC float3 make_float3(int3 a);
HOST_FUNC DEV_FUNC float3 make_float3(float4 a);
HOST_FUNC DEV_FUNC float4 make_float4(float a);
HOST_FUNC DEV_FUNC float4 make_float4(float3 a);
HOST_FUNC DEV_FUNC float4 make_float4(float3 a, float b);

/* ---------------- *
 * make_double(num) *
 *----------------- */
HOST_FUNC DEV_FUNC double4 make_double4(double a);

/* ----------------- *
 * operator+(type)   *
 *------------------ */
HOST_FUNC DEV_FUNC int3 operator+(int3 a, int3 b);
HOST_FUNC DEV_FUNC int2 operator+(int2 a, int2 b);
HOST_FUNC DEV_FUNC float2 operator+(float2 a, float2 b);
HOST_FUNC DEV_FUNC uchar3 operator+(uchar3 a, uchar3 b);
HOST_FUNC DEV_FUNC float3 operator+(float3 a, float b);
HOST_FUNC DEV_FUNC float3 operator+(float3 a, float3 b);
HOST_FUNC DEV_FUNC float4 operator+(float4 a, float4 b);

/* ----------------- *
 * operator+=(type)  *
 *------------------ */
HOST_FUNC DEV_FUNC void operator+=(float3 & a, uchar3 b);
HOST_FUNC DEV_FUNC void operator+=(float3 & a, float3 b);
HOST_FUNC DEV_FUNC void operator-=(float3 & a, float3 b);

/* ----------------- *
 * operator-(type)   *
 *------------------ */
HOST_FUNC DEV_FUNC uchar3 operator-(uchar3 a, uchar3 b);
HOST_FUNC DEV_FUNC int2 operator-(int2 a, int2 b);
HOST_FUNC DEV_FUNC float3 operator-(float3 b);
HOST_FUNC DEV_FUNC float3 operator-(float3 a, float b);
HOST_FUNC DEV_FUNC float3 operator-(float a, float3 b);
HOST_FUNC DEV_FUNC float3 operator-(float3 a, float3 b);
HOST_FUNC DEV_FUNC float4 operator-(float4 a, float4 b);

/* ------------------- *
 * operator(vec)*(vec) *
 *-------------------- */
HOST_FUNC DEV_FUNC float operator*(float3 a, float3 b);
HOST_FUNC DEV_FUNC float operator*(float3 a, float4 b);
HOST_FUNC DEV_FUNC float operator*(float4 a, float3 b);
HOST_FUNC DEV_FUNC float operator*(float4 a, float4 b);

/* ---------------------- *
 * operator(vec)*(scalar) *
 *----------------------- */
HOST_FUNC DEV_FUNC uchar3 operator*(uchar3 a, ushort b);
HOST_FUNC DEV_FUNC uchar3 operator*(uchar3 a, int b);
HOST_FUNC DEV_FUNC uchar3 operator*(int b, uchar3 a);
HOST_FUNC DEV_FUNC int3 operator*(int3 a, uint b);
HOST_FUNC DEV_FUNC int3 operator*(int3 a, int b);
HOST_FUNC DEV_FUNC int3 operator*(float3 a, int b);
HOST_FUNC DEV_FUNC float3 operator*(int3 a, float b);
HOST_FUNC DEV_FUNC float3 operator*(float3 a, float b);
HOST_FUNC DEV_FUNC float3 operator*(float a, float3 b);
HOST_FUNC DEV_FUNC float4 operator*(float4 a, float b);

/* ------------------- *
 * operator(vec)/(vec) *
 *-------------------- */
HOST_FUNC DEV_FUNC int3 operator/(int3 a, int3 b);
HOST_FUNC DEV_FUNC float3 operator/(float3 a, int3 b);
HOST_FUNC DEV_FUNC float3 operator/(float3 a, float3 b);
HOST_FUNC DEV_FUNC float4 operator/(float4 a, float4 b);

/* ---------------------- *
 * operator(vec)/(scalar) *
 *----------------------- */
HOST_FUNC DEV_FUNC int3 operator/(int3 a, int b);
HOST_FUNC DEV_FUNC int2 operator/(int2 a, int b);
HOST_FUNC DEV_FUNC int3 operator/(int3 a, uint b);
HOST_FUNC DEV_FUNC float2 operator/(float2 a, int b);
HOST_FUNC DEV_FUNC uchar3 operator/(uchar3 a, int b);
HOST_FUNC DEV_FUNC float3 operator/(float3 a, int b);
HOST_FUNC DEV_FUNC float3 operator/(float3 a, float b);
HOST_FUNC DEV_FUNC float3 operator/(float a, float3 b);
HOST_FUNC DEV_FUNC float4 operator/(float4 a, float b);

/* ---------------------- *
 * operator(vec)%(scalar) *
 *----------------------- */
HOST_FUNC DEV_FUNC int3 operator%(int3 a, int b);

/* ---------------------- *
 * operator(vec) == (vec) *
 *----------------------- */
HOST_FUNC DEV_FUNC bool operator==(int3 a, int3 b);

/* ---------------- *
 * cross(type(num)) *
 *----------------- */
HOST_FUNC DEV_FUNC float3 cross(float3 a, float3 b);
HOST_FUNC DEV_FUNC float3 cross(float4 a, float4 b);

/* ---------------- *
 * max(float(num))  *
 *----------------- */
DEV_FUNC void atomicMax(float* address, float val);

/* ---------------- *
 * min(float(num))  *
 *----------------- */
DEV_FUNC void atomicMin(float* address, float val);

/* ------------ *
 *  norm(type)  *
 *------------- */
DEV_FUNC float norm(float3 a);
DEV_FUNC float norm(float4 a);

/* -------------------- *
 *  inverse norm(type)  *
 *--------------------- */
DEV_FUNC float inv_norm(float3 a);
DEV_FUNC float inv_norm(float4 a);

/* ----------------- *
 *  Normalised(type) *
 *------------------ */
DEV_FUNC float3 normalised(float3 a);
DEV_FUNC float4 normalised(float4 a);

/*------------- *
 *  floor(vec)  *
 *------------- */
DEV_FUNC float3 floor(float3 a);

/*------------- *
 *   Matrix3f   *
 *------------- */
struct Matrix3f {
	HOST_FUNC DEV_FUNC Matrix3f();
	HOST_FUNC DEV_FUNC static Matrix3f Identity();
	HOST_FUNC DEV_FUNC float3 operator*(float3 vec);
	HOST_FUNC DEV_FUNC float4 operator*(float4 vec);
	HOST_FUNC DEV_FUNC float3 operator*(float3 vec) const;
	HOST_FUNC DEV_FUNC float4 operator*(float4 vec) const;

	float3 rowx;
	float3 rowy;
	float3 rowz;
};

#endif /* VECMATH_H_ */

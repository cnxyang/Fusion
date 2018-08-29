#include "device_math.hpp"

DEV_FUNC uchar3 make_uchar3(float3 a) {
	return make_uchar3(__float2int_rd(a.x),
					   __float2int_rd(a.y),
					   __float2int_rd(a.z));
}

HOST_FUNC DEV_FUNC int2 make_int2(int a) {
	return make_int2(a, a);
}

DEV_FUNC int2 make_int2(float2 a) {
	return make_int2(__float2int_rd(a.x),
				     __float2int_rd(a.y));
}

HOST_FUNC DEV_FUNC int3 make_int3(int a) {
	return make_int3(a, a, a);
}

DEV_FUNC int3 make_int3(float a) {
	return make_int3(make_float3(a));
}

DEV_FUNC int3 make_int3(float3 a) {
	return make_int3(__float2int_rd(a.x),
					 __float2int_rd(a.y),
					 __float2int_rd(a.z));
}

HOST_FUNC DEV_FUNC int4 make_int4(int3 a, int b) {
	return make_int4(a.x, a.y, a.z, b);
}


HOST_FUNC DEV_FUNC uint2 make_uint2(int a) {
	return make_uint2(a, a);
}

HOST_FUNC DEV_FUNC uint3 make_uint3(int a) {
	return make_uint3(a, a, a);
}

HOST_FUNC DEV_FUNC float2 make_float2(float a) {
	return make_float2(a, a);
}

HOST_FUNC DEV_FUNC float3 make_float3(uchar3 a) {
	return make_float3(a.x, a.y, a.z);
}

HOST_FUNC DEV_FUNC float3 make_float3(float a) {
	return make_float3(a, a, a);
}

HOST_FUNC DEV_FUNC float3 make_float3(int3 a) {
	return make_float3(a.x, a.y, a.z);
}

HOST_FUNC DEV_FUNC float3 make_float3(float4 a) {
	return make_float3(a.x, a.y, a.z);
}

HOST_FUNC DEV_FUNC float4 make_float4(float a) {
	return make_float4(a, a, a, a);
}

HOST_FUNC DEV_FUNC float4 make_float4(float3 a) {
	return make_float4(a.x, a.y, a.z, 1.f);
}

HOST_FUNC DEV_FUNC float4 make_float4(float3 a, float b) {
	return make_float4(a.x, a.y, a.z, b);
}

HOST_FUNC DEV_FUNC double4 make_double4(double a) {
	return make_double4(a, a, a, a);
}

HOST_FUNC DEV_FUNC int2 operator+(int2 a, int2 b) {
	return make_int2(a.x + b.x, a.y + b.y);
}

HOST_FUNC DEV_FUNC float2 operator+(float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

HOST_FUNC DEV_FUNC uchar3 operator+(uchar3 a, uchar3 b) {
	return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_FUNC DEV_FUNC int3 operator+(int3 a, int3 b) {
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_FUNC DEV_FUNC float3 operator+(float3 a, float b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

HOST_FUNC DEV_FUNC float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_FUNC DEV_FUNC float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

HOST_FUNC DEV_FUNC void operator+=(float3 & a, uchar3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

HOST_FUNC DEV_FUNC void operator+=(float3 & a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

HOST_FUNC DEV_FUNC void operator-=(float3 & a, float3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

HOST_FUNC DEV_FUNC int2 operator-(int2 a, int2 b) {
	return make_int2(a.x - b.x, a.y - b.y);
}

HOST_FUNC DEV_FUNC float3 operator-(float3 b) {
	return make_float3(-b.x, -b.y, -b.z);
}

HOST_FUNC DEV_FUNC float3 operator-(float3 a, float b) {
	return make_float3(a.x - b, a.y - b, a.z - b);
}

HOST_FUNC DEV_FUNC float3 operator-(float a, float3 b) {
	return make_float3(a - b.x, a - b.y, a - b.z);
}

HOST_FUNC DEV_FUNC float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_FUNC DEV_FUNC float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

HOST_FUNC DEV_FUNC
float operator*(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_FUNC DEV_FUNC
float operator*(float3 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_FUNC DEV_FUNC
float operator*(float4 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w;
}

HOST_FUNC DEV_FUNC
float operator*(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

HOST_FUNC DEV_FUNC uchar3 operator*(uchar3 a, unsigned short b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC uchar3 operator*(uchar3 a, int b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC uchar3 operator*(int b, uchar3 a) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC int3 operator*(int3 a, uint b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC int3 operator*(int3 a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC int3 operator*(float3 a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC float3 operator*(int3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

HOST_FUNC DEV_FUNC float3 operator*(float a, float3 b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

HOST_FUNC DEV_FUNC float4 operator*(float4 a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

HOST_FUNC DEV_FUNC int3 operator/(int3 a, int3 b) {
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HOST_FUNC DEV_FUNC float3 operator/(float3 a, int3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HOST_FUNC DEV_FUNC float3 operator/(float3 a, float3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HOST_FUNC DEV_FUNC float4 operator/(float4 a, float4 b) {
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}


HOST_FUNC DEV_FUNC int2 operator/(int2 a, int b) {
	return make_int2(a.x / b, a.y / b);
}

HOST_FUNC DEV_FUNC float2 operator/(float2 a, int b) {
	return make_float2(a.x / b, a.y / b);
}

HOST_FUNC DEV_FUNC uchar3 operator/(uchar3 a, int b) {
	return make_uchar3(a.x / b, a.y / b, a.z / b);
}

HOST_FUNC DEV_FUNC int3 operator/(int3 a, uint b) {
	return make_int3(a.x / (int) b, a.y / (int) b, a.z / (int) b);
}

HOST_FUNC DEV_FUNC int3 operator/(int3 a, int b) {
	return make_int3(a.x / b, a.y / b, a.z / b);
}

HOST_FUNC DEV_FUNC float3 operator/(float3 a, int b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

HOST_FUNC DEV_FUNC float3 operator/(float3 a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

HOST_FUNC DEV_FUNC float3 operator/(float a, float3 b) {
	return make_float3(a / b.x, a / b.y, a / b.z);
}

HOST_FUNC DEV_FUNC float4 operator/(float4 a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

HOST_FUNC DEV_FUNC int3 operator%(int3 a, int b) {
	return make_int3(a.x % b, a.y % b, a.z % b);
}

HOST_FUNC DEV_FUNC
bool operator==(int3 a, int3 b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

DEV_FUNC float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

DEV_FUNC float3 cross(float4 a, float4 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

DEV_FUNC void atomicMax(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i,
				assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

DEV_FUNC void atomicMin(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i,
				assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

DEV_FUNC
float norm(float3 a) {
	return sqrtf(a * a);
}

DEV_FUNC
float norm(float4 a) {
	return sqrtf(a * a);
}

DEV_FUNC
float inv_norm(float3 a) {
	return rsqrtf(a * a);
}

DEV_FUNC
float inv_norm(float4 a) {
	return rsqrtf(a * a);
}

DEV_FUNC float3 normalised(float3 a) {
	return a / norm(a);
}

DEV_FUNC float4 normalised(float4 a) {
	return a / norm(a);
}

DEV_FUNC float3 floor(float3 a) {
	return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

HOST_FUNC DEV_FUNC Matrix3f::Matrix3f() {
	rowx = rowy = rowz = make_float3(0, 0, 0);
}

HOST_FUNC DEV_FUNC Matrix3f Matrix3f::Identity() {
	Matrix3f id;
	id.rowx = make_float3(1, 0, 0);
	id.rowy = make_float3(0, 1, 0);
	id.rowz = make_float3(0, 0, 1);
	return id;
}

HOST_FUNC DEV_FUNC float3 Matrix3f::operator*(float3 vec) {
	return make_float3(rowx * vec, rowy * vec, rowz * vec);
}

HOST_FUNC DEV_FUNC float3 Matrix3f::operator*(float3 vec) const {
	return make_float3(rowx * vec, rowy * vec, rowz * vec);
}

HOST_FUNC DEV_FUNC float4 Matrix3f::operator*(float4 vec) {
	return make_float4(rowx * vec, rowy * vec, rowz * vec, vec.w);
}

HOST_FUNC DEV_FUNC float4 Matrix3f::operator*(float4 vec) const {
	return make_float4(rowx * vec, rowy * vec, rowz * vec, vec.w);
}

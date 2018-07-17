#include "DeviceArray.h"
#include "DeviceMath.h"
#include "DeviceFunc.h"
#include <curand_kernel.h>

#define RANSAC_MAX 100
#define MAXKP 1000

__device__ inline
int clamp(int n, int minval, int maxval) {
	return max(min(n, maxval), minval);
}

__global__ void
FindTransform_device(const PtrSz<Point> srcpts,
									   const PtrSz<Point> dstpts) {

    curandState state;
    curand_init(clock64(), threadIdx.x, 0, &state);
    int numbers[3] = { clamp((int)(curand_uniform(&state) * srcpts.size), 0, srcpts.size),
    							       clamp((int)(curand_uniform(&state) * srcpts.size), 0, srcpts.size),
    							       clamp((int)(curand_uniform(&state) * srcpts.size), 0, srcpts.size) };

    const Point& a = srcpts[numbers[0]];
    const Point& b = srcpts[numbers[1]];
    const Point& c = srcpts[numbers[2]];
//    printf("a.ptr: %d\n, b.ptr: %d\n, c.ptr: %d\n", srcpts[0].ptr, srcpts[1].ptr, srcpts[2].ptr);

    if(!a.valid || !b.valid || !c.valid ||
    	a.ptr < 0 || b.ptr < 0 || c.ptr < 0 ||
        a.ptr >= dstpts.size || b.ptr >= dstpts.size || c.ptr >= dstpts.size)
    	return;

    Point A = dstpts[a.ptr];
    Point B(dstpts[b.ptr]);
    Point C(dstpts[c.ptr]);
    if(!A.valid || !B.valid || !C.valid )
    	return;
//    printf("A: %f, %f, %f\n", A.pos.x, A.pos.y, A.pos.z);
}

void FindTransform(const Frame& frame, const DeviceArray<Point>& mapPoint) {

	dim3 block(RANSAC_MAX);

//	FindTransform_device<<<1, block>>>(frame.mMapPoints, mapPoint);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__device__ inline int
HammingDist(const char& a,const char& b) {
	return __popc(a ^ b);
}

__device__ inline int
DescriptorDist(const char* a,const char* b) {
	int dist = 0;

#pragma unroll
	for(int i = 0; i < 8; ++i) {
		dist += HammingDist(a[i], b[i]);
	}
	return dist;
}

__global__ void
MatchKeyPoint_device(//PtrSz<Point> srcpts,
									   const PtrStep<char> srcdes,
//									   const PtrSz<Point> dstpts,
									   const PtrStep<char> dstdes,
									   PtrSz<int> index) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= index.size)
		return;

	int dist = 64;
	int id = -1;
	const char* src_des = srcdes.ptr(idx);
	const int size = index.size;
//	srcpts[idx].ptr = -1;

//	if(!isnan(srcpts[idx].pos.x) && srcpts[idx].valid) {
	{
		for(int i = 0; i < size; ++i) {
//			if(dstpts[i].valid) {
			{
				int tmp = DescriptorDist(src_des, dstdes.ptr(i));
				if(tmp < dist) {
					dist = tmp;
					id = i;
				}
			}
		}
	}
	index[idx] = id;
}

bool MatchKeyPoint(const Frame& frame, const DeviceArray<Point>& mapPoint, const DeviceArray2D<char>& Des) {
	if(frame.mMapPoints.empty() || mapPoint.empty())
		return false;

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(frame.mMapPoints.size(), block.x));

	DeviceArray<int> index(frame.mMapPoints.size());
//	MatchKeyPoint_device<<<grid, block>>>(frame.mDescriptors, Des, index);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	return true;
}

#define VoxelSize 0.005
#define NoBucket 20000;

__device__ inline
int HashKey(const float3 pt) {
	int3 pos = make_int3(pt / (float)VoxelSize);
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % NoBucket;
	if(res < 0) res += NoBucket;
		return res;
}

__device__ inline
void Swap(int2& a, int2& b) {
	a = a + b - (b = a);
}

__global__ void
BitonicSort_device(PtrSz<int2> src) {
	// TODO: implementation
}

__global__ void
GenerateIndexArray_device(const PtrSz<Point> src, PtrSz<int2> index) {

	const int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id >= src.size)
		return;

	const Point& pt = src[id];
	if(pt.valid) {
		int key = HashKey(pt.pos);
		index[id] = make_int2(key, id);
	}
	else {
		index[id] = make_int2(-1);
	}
}

void GenerateIndexArray(const DeviceArray<Point>& src, DeviceArray<int2>& index, int N) {

	dim3 block(1024);
	dim3 grid(cv::divUp(N, block.x));

	GenerateIndexArray_device<<<grid, block>>>(src, index);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
AppendMapPoints_device(const PtrSz<Point> src, const PtrStep<char> des_src,
											PtrSz<Point> dst, PtrStep<char> des_dst,
											int start, int end) {

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index + start > end || index >= src.size)
		return;

	dst[index + start] = src[index];
	for(int i = 0; i < 32; ++i) {
		des_dst.ptr(index + start)[i] = des_src.ptr(index)[i];
	}
}

void AppendMapPoints(const DeviceArray<Point>& src, const DeviceArray2D<char>& des_src,
										DeviceArray<Point>& dst, DeviceArray2D<char>& des_dst,
										int start, int end) {

	if(end < start)
		return;

	dim3 block(1024);
	dim3 grid(cv::divUp(end - start + 1, block.x));

	AppendMapPoints_device<<<grid, block>>>(src, des_src, dst, des_dst, start, end);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

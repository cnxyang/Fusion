#include "Map.h"

__device__ MapDesc pDesc;

void Map::UpdateDesc(MapDesc& desc) {
	memcpy(&mDesc, &desc, sizeof(MapDesc));
	SafeCall(cudaMemcpyToSymbol(pDesc, &mDesc, sizeof(MapDesc)));
}

void Map::DownloadDesc() {
	SafeCall(cudaMemcpyFromSymbol(&mDesc, pDesc, sizeof(MapDesc)));
}

__global__ void
FuseKeyPoints_device(DeviceMap map, PtrSz<Point> pts, PtrStep<char> descriptors) {

}

void Map::FuseKeyPoints(const Frame& frame) {
	static bool firstcall = false;
}

__device__ int
HammingDist(const char& a,const char& b) {
	return __popc(a ^ b);
}

__device__ int
DescriptorDist(const char* a,const char* b) {
	int dist = 0;
	for(int i = 0; i < 8; ++i) {
		dist += HammingDist(a[i], b[i]);
	}
	return dist;
}

__global__ void
MatchKeyPoint_device(const DeviceMap map,
									   const PtrSz<Point> pts,
									   PtrStep<char> descriptors,
									   PtrSz<Point> match) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= pts.size)
		return;

	int dist = 256;
	const int HammingThresh = 64;
	const char* descriptor = descriptors.ptr(idx);
	Point& result = match[idx];
	result.valid = false;
	for(int i = 0; i < map.keyPoints.size; ++i) {
		Point kp = map.keyPoints[i];
		if(kp.valid) {
			int tmp = DescriptorDist(descriptor, map.descriptors.ptr(i));
			if(tmp < dist) {
				dist = tmp;
				result = kp;
			}

			if(tmp < HammingThresh)
				return;
		}
	}
}

bool Map::MatchKeyPoint(const Frame& frame, int k) {
	if(frame.mMapPoints.empty())
		return false;

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(frame.mMapPoints.size(), block.x));

	DeviceArray<Point> match(frame.mMapPoints.size());
	MatchKeyPoint_device<<<grid, block>>>(*this, frame.mMapPoints, frame.mDescriptors, match);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	return true;
}

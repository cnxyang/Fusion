#include "Timer.hpp"
#include "device_mapping.cuh"

__global__ void CollectORBKeys(KeyMap Km, PtrSz<ORBKey> index, int* totalKeys) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= Km.Keys.size)
		return;

	ORBKey* key = &Km.Keys[idx];

	if (key->valid) {
		int id = atomicAdd(totalKeys, 1);
		memcpy((void*) &index[id], (void*) key, sizeof(ORBKey));
	}
}

void CollectKeys(KeyMap Km, DeviceArray<ORBKey>& keys, int& n) {

	keys.create(Km.Keys.size);

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(Km.Keys.size, block.x));

	DeviceArray<int> totalKeys(1);
	totalKeys.zero();

	CollectORBKeys<<<grid, block>>>(Km, keys, totalKeys);

	totalKeys.download(&n);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void InsertKeysKernel(KeyMap map, PtrSz<ORBKey> key) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= key.size)
		return;

	map.InsertKey(&key[idx]);
}

void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys) {
	if(keys.size() == 0)
		return;

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(keys.size(), block.x));

	InsertKeysKernel<<<grid, block>>>(map, keys);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResetKeysKernel(KeyMap map) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	map.ResetKeys(idx);
}

void ResetKeys(KeyMap map) {
	dim3 block(MaxThread);
	dim3 grid(cv::divUp(map.MaxKeys * map.nBuckets, block.x));

	ResetKeysKernel<<<grid, block>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

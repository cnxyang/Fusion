#include "device_mapping.cuh"

//__global__ void CollectORBKeys(KeyMap Km, PtrSz<int> index, int* totalKeys) {
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	if(idx >= Km.Keys.size)
//		return;
//
//	ORBKey* key = &Km.Keys[idx];
//	if(key->valid) {
//		int id = atomicAdd(totalKeys, 1);
//		index[id] = idx;
//	}
//}

//void CollectORBKeys(KeyMap Km, PtrSz<int> index, int* totalKeys) {
//
//}

#include "Timer.hpp"
#include "device_mapping.cuh"

#define CUDA_KERNEL __global__

CUDA_KERNEL void CollectORBKeys(KeyMap Km, PtrSz<ORBKey> index, int* totalKeys) {
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

CUDA_KERNEL void InsertKeysKernel(KeyMap map, PtrSz<ORBKey> key) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= key.size)
		return;

	map.InsertKey(&key[idx]);
}

void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys) {
	if (keys.size() == 0)
		return;

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(keys.size(), block.x));

	InsertKeysKernel<<<grid, block>>>(map, keys);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

CUDA_KERNEL void ResetKeysKernel(KeyMap map) {
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

CUDA_KERNEL void BuildAdjecencyMatrixKernel(cv::cuda::PtrStepSz<float> AM,
		PtrSz<ORBKey> TrainKeys, PtrSz<ORBKey> QueryKeys,
		PtrSz<float> MatchDist) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < AM.cols && y < AM.rows) {
		float score = 0;
		if(x == y) {
			score = expf(-MatchDist[x]);
		}
		else {
			ORBKey* match_0_train = &TrainKeys[x];
			ORBKey* match_0_query = &QueryKeys[x];
			ORBKey* match_1_train = &TrainKeys[y];
			ORBKey* match_1_query = &QueryKeys[y];
			float d_0 = norm(match_0_train->pos - match_0_query->pos);
			float d_1 = norm(match_1_train->pos - match_1_query->pos);
			if(d_0 > 1e-6 && d_1 > 1e-6) {
				float alpha_0 = acosf(match_0_train->normal * match_0_query->normal);
				float alpha_1 = acosf(match_1_train->normal * match_1_query->normal);
				float beta_0 = acosf(match_0_train->normal * (match_0_query->pos - match_0_train->pos));
				float beta_1 = acosf(match_1_train->normal * (match_1_query->pos - match_1_train->pos));
				float gamma_0 = acosf(match_0_query->normal * (match_0_train->pos - match_0_query->pos));
				float gamma_1 = acosf(match_1_query->normal * (match_1_train->pos - match_1_query->pos));
				score = expf(-(fabs(d_0 - d_1) + fabs(alpha_0 - alpha_1) + fabs(beta_0 - beta_1) + fabs(gamma_0 - gamma_1)));
			}
		}
		if(isnan(score))
			score = 0;
		AM.ptr(y)[x] = score;
	}
}

CUDA_KERNEL void SelectMatches(PtrSz<ORBKey> TrainKeys, PtrSz<ORBKey> QueryKeys,
		PtrSz<ORBKey> TrainKeys_selected, PtrSz<ORBKey> QueryKeys_selected,
		PtrSz<int> SelectedMatches, PtrSz<int> QueryIdx, PtrSz<int> SelectedIdx) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < SelectedMatches.size) {
		int idx = SelectedMatches[x];
		ORBKey* trainKey = &TrainKeys[idx];
		ORBKey* queryKey = &QueryKeys[idx];
		memcpy((void*) &TrainKeys_selected[x], (void*) trainKey, sizeof(ORBKey));
		memcpy((void*) &QueryKeys_selected[x], (void*) queryKey, sizeof(ORBKey));
		SelectedIdx[x] = QueryIdx[idx];
	}
}

void BuildAdjecencyMatrix(cv::cuda::GpuMat& AM,	DeviceArray<ORBKey>& TrainKeys,
		DeviceArray<ORBKey>& QueryKeys, DeviceArray<float>& MatchDist,
		DeviceArray<ORBKey>& train_select, DeviceArray<ORBKey>& query_select,
		DeviceArray<int>& QueryIdx, DeviceArray<int>& SelectedIdx) {

	dim3 block(32, 8);
	dim3 grid(cv::divUp(AM.cols, block.x), cv::divUp(AM.rows, block.y));

	BuildAdjecencyMatrixKernel<<<grid, block>>>(AM, TrainKeys, QueryKeys,
			MatchDist);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::cuda::GpuMat result;
	cv::cuda::reduce(AM, result, 0, CV_REDUCE_SUM);
	cv::Mat cpuResult, indexMat;
	result.download(cpuResult);

	cv::sortIdx(cpuResult, indexMat, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	int selection = indexMat.cols >= 200 ? 200 : indexMat.cols;
	DeviceArray<int> SelectedMatches(selection);
	SelectedMatches.upload((void*)indexMat.data, selection);
	train_select.create(selection);
	query_select.create(selection);
	SelectedIdx.create(selection);

	SelectMatches<<<1, selection>>>(TrainKeys, QueryKeys, train_select,
			query_select, SelectedMatches, QueryIdx, SelectedIdx);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

#include "Reduction.h"
#include "RenderScene.h"
#include "ParallelScan.h"

__global__ void CollectORBKeys(KeyMap Km, PtrSz<ORBKey> keys, PtrSz<int> index, uint* totalKeys) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ bool scan;
	if(idx == 0)
		scan = false;
	__syncthreads();
	uint val = 0;
	if (idx < Km.Keys.size) {
		ORBKey* key = &Km.Keys[idx];
		if (key->valid && key->obs > 0) {
			scan = true;
			val = 1;
		}
	}
	__syncthreads();
	if(scan) {
		int offset = ComputeOffset<1024>(val, totalKeys);
		if(offset >= 0) {
			memcpy((void*) &keys[offset], (void*) &Km.Keys[idx], sizeof(ORBKey));
			index[offset] = idx;
		}
	}
}

void CollectKeys(KeyMap Km, DeviceArray<ORBKey>& keys, DeviceArray<int> & index, uint& n) {

	keys.create(Km.Keys.size);

	dim3 block(MaxThread);
	dim3 grid(DivUp(Km.Keys.size, block.x));

	DeviceArray<uint> totalKeys(1);
	totalKeys.clear();

	CollectORBKeys<<<grid, block>>>(Km, keys, index, totalKeys);

	totalKeys.download(&n);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void InsertKeysKernel(KeyMap map, PtrSz<ORBKey> key, PtrSz<int> indices) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < key.size) {
		map.InsertKey(&key[idx], indices[idx]);
	}
}

void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys, DeviceArray<int> & indices) {
	if (keys.size == 0)
		return;

	dim3 block(MaxThread);
	dim3 grid(DivUp(keys.size, block.x));

	InsertKeysKernel<<<grid, block>>>(map, keys, indices);

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

__global__ void BuildAdjecencyMatrixKernel(cv::cuda::PtrStepSz<float> AM,
		PtrSz<ORBKey> TrainKeys, PtrSz<ORBKey> QueryKeys,
		PtrSz<float> MatchDist) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < AM.cols && y < AM.rows) {
		float score = 0;
		if(x == y) {
			score = expf(-MatchDist[x]);
		} else {
			ORBKey* match_0_train = &TrainKeys[x];
			ORBKey* match_0_query = &QueryKeys[x];
			ORBKey* match_1_train = &TrainKeys[y];
			ORBKey* match_1_query = &QueryKeys[y];
			float d_0 = norm(match_0_train->pos - match_1_train->pos);
			float d_1 = norm(match_0_query->pos - match_1_query->pos);
			if(d_0 > 1e-6 && d_1 > 1e-6) {
				float alpha_0 = acosf(match_0_train->normal * match_1_train->normal);
				float alpha_1 = acosf(match_0_query->normal * match_1_query->normal);
				float beta_0 = acosf(match_0_train->normal * (match_1_train->pos - match_0_train->pos) / d_0);
				float beta_1 = acosf(match_1_train->normal * (match_1_train->pos - match_0_train->pos) / d_0);
				float gamma_0 = acosf(match_0_query->normal * (match_1_query->pos - match_0_query->pos) / d_1);
				float gamma_1 = acosf(match_1_query->normal * (match_1_query->pos - match_0_query->pos) / d_1);
				score = expf(-(fabs(d_0 - d_1) + fabs(alpha_0 - alpha_1) + fabs(beta_0 - beta_1) + fabs(gamma_0 - gamma_1)));
//				score = expf(-(fabs(d_0 - d_1) + fabs(alpha_0 - alpha_1)));
			}
		}
		if(isnan(score))
			score = 0;
		AM.ptr(y)[x] = score;
	}
}

__global__ void SelectMatches(PtrSz<ORBKey> TrainKeys, PtrSz<ORBKey> QueryKeys,
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
	dim3 grid(DivUp(AM.cols, block.x), DivUp(AM.rows, block.y));

	BuildAdjecencyMatrixKernel<<<grid, block>>>(AM, TrainKeys, QueryKeys, MatchDist);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::cuda::GpuMat result;
	cv::cuda::reduce(AM, result, 0, CV_REDUCE_SUM);
	cv::Mat cpuResult, indexMat;
	result.download(cpuResult);

	cv::sortIdx(cpuResult, indexMat, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	int selection = indexMat.cols >= 100 ? 100 : indexMat.cols;
	DeviceArray<int> SelectedMatches(selection);
	SelectedMatches.upload((void*)indexMat.data, selection);
	train_select.create(selection);
	query_select.create(selection);
	SelectedIdx.create(selection);

	dim3 block2(MaxThread);
	dim3 grid2(DivUp(selection, block2.x));

	SelectMatches<<<grid2, block2>>>(TrainKeys, QueryKeys, train_select,
			query_select, SelectedMatches, QueryIdx, SelectedIdx);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ProjectVisibleKeysKernel(KeyMap map, Matrix3f invRot, float3 trans,
		int cols, int rows, float maxd, float mind, float fx, float fy,
		float cx, float cy) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x < KeyMap::MaxKeys * KeyMap::nBuckets) {
		ORBKey& key = map.Keys[x];
		if(key.valid && key.obs <= 8) {
			float3 pos = invRot * (key.pos - trans);
			float u = fx * pos.x / pos.z + cx;
			float v = fy * pos.y / pos.z + cy;
			if(u >= 0 && v >= 0 && u < cols && v < rows
					&& pos.z < maxd && pos.z > mind) {
				key.obs -= 1;
				if(key.obs <= KeyMap::MinObsThresh) {
					key.valid = false;
				}
			}
		}
	}
}

void ProjectVisibleKeys(KeyMap map, Matrix3f RviewInv, float3 tview, int cols,
		int rows, float fx, float fy, float cx, float cy) {

	dim3 block(MaxThread);
	dim3 grid(DivUp(map.Keys.size, block.x));

	ProjectVisibleKeysKernel<<<grid, block>>>(map, RviewInv, tview, cols, rows,
			DeviceMap::DepthMax, DeviceMap::DepthMin, fx, fy, cx, cy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

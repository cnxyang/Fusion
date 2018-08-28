#include "Timer.hpp"
#include "device_mapping.cuh"

#define CUDA_KERNEL __global__

template<int threadBlock>
DEV_FUNC int ComputeOffset(uint element, uint *sum) {

	__shared__ uint buffer[threadBlock];
	__shared__ uint blockOffset;

	if (threadIdx.x == 0)
		memset(buffer, 0, sizeof(uint) * 16 * 16);
	__syncthreads();

	buffer[threadIdx.x] = element;
	__syncthreads();

	int s1, s2;

	for (s1 = 1, s2 = 1; s1 < threadBlock; s1 <<= 1) {
		s2 |= s1;
		if ((threadIdx.x & s2) == s2)
			buffer[threadIdx.x] += buffer[threadIdx.x - s1];
		__syncthreads();
	}

	for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1) {
		if (threadIdx.x != threadBlock - 1 && (threadIdx.x & s2) == s2)
			buffer[threadIdx.x + s1] += buffer[threadIdx.x];
		__syncthreads();
	}

	if (threadIdx.x == 0 && buffer[threadBlock - 1] > 0)
		blockOffset = atomicAdd(sum, buffer[threadBlock - 1]);
	__syncthreads();

	int offset;
	if (threadIdx.x == 0) {
		if (buffer[threadIdx.x] == 0)
			offset = -1;
		else
			offset = blockOffset;
	} else {
		if (buffer[threadIdx.x] == buffer[threadIdx.x - 1])
			offset = -1;
		else
			offset = blockOffset + buffer[threadIdx.x - 1];
	}

	return offset;
}

CUDA_KERNEL void CollectORBKeys(KeyMap Km,
		PtrSz<ORBKey> index, uint* totalKeys) {
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
			memcpy((void*) &index[offset], (void*) &Km.Keys[idx], sizeof(ORBKey));
		}
	}
}

void CollectKeys(KeyMap Km, DeviceArray<ORBKey>& keys, uint& n) {

	keys.create(Km.Keys.size);

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(Km.Keys.size, block.x));

	DeviceArray<uint> totalKeys(1);
	totalKeys.zero();

	CollectORBKeys<<<grid, block>>>(Km, keys, totalKeys);

	totalKeys.download(&n);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

CUDA_KERNEL void InsertKeysKernel(KeyMap map, PtrSz<ORBKey> key) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < key.size) {
		map.InsertKey(&key[idx]);
	}
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
			float d_0 = norm(match_0_train->pos - match_1_train->pos);
			float d_1 = norm(match_0_query->pos - match_1_query->pos);
			if(d_0 > 1e-6 && d_1 > 1e-6) {
				float alpha_0 = acosf(match_0_train->normal * match_1_train->normal);
				float alpha_1 = acosf(match_0_query->normal * match_1_query->normal);
//				float beta_0 = acosf(match_0_train->normal * (match_1_train->pos - match_0_train->pos) / d_0);
//				float beta_1 = acosf(match_1_train->normal * (match_1_train->pos - match_0_train->pos) / d_0);
//				float gamma_0 = acosf(match_0_query->normal * (match_1_query->pos - match_0_query->pos) / d_1);
//				float gamma_1 = acosf(match_1_query->normal * (match_1_query->pos - match_0_query->pos) / d_1);
//				score = expf(-(fabs(d_0 - d_1) + fabs(alpha_0 - alpha_1) + fabs(beta_0 - beta_1) + fabs(gamma_0 - gamma_1)));
				score = expf(-(fabs(d_0 - d_1) + fabs(alpha_0 - alpha_1)));
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
	int selection = indexMat.cols >= 100 ? 100 : indexMat.cols;
	DeviceArray<int> SelectedMatches(selection);
	SelectedMatches.upload((void*)indexMat.data, selection);
	train_select.create(selection);
	query_select.create(selection);
	SelectedIdx.create(selection);

	dim3 block2(MaxThread);
	dim3 grid2(cv::divUp(selection, block2.x));

	SelectMatches<<<grid2, block2>>>(TrainKeys, QueryKeys, train_select,
			query_select, SelectedMatches, QueryIdx, SelectedIdx);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

CUDA_KERNEL void ProjectVisibleKeysKernel(KeyMap map, Matrix3f invRot, float3 trans,
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

void ProjectVisibleKeys(KeyMap map, Frame& F) {

	dim3 block(MaxThread);
	dim3 grid(cv::divUp(map.Keys.size, block.x));

	ProjectVisibleKeysKernel<<<grid, block>>>(map, F.RotInv_gpu(),
			F.Trans_gpu(), Frame::cols(0), Frame::rows(0), DeviceMap::DepthMax,
			DeviceMap::DepthMin, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0));

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

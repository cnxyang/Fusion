#include "DeviceMap.h"

#include <opencv.hpp>
#include <cudaarithm.hpp>

__device__ __forceinline__ float clamp(float a, float min = -1, float max = 1) {
	a = a > min ? a : min;
	a = a < max ? a : max;
	return a;
}

__global__ void BuildAdjecencyMatrixKernel(
		cv::cuda::PtrStepSz<float> adjecencyMatrix, PtrSz<SURF> frameKeys,
		PtrSz<SURF> mapKeys, PtrSz<float> dist) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= adjecencyMatrix.cols || y >= adjecencyMatrix.rows)
		return;

	float score = 0;
	if(x == y) {
		score = exp(-dist[x]);
	} else {

		SURF * mapKey00 = &mapKeys[x];
		SURF * mapKey01 = &mapKeys[y];

		SURF * frameKey00 = &frameKeys[x];
		SURF * frameKey01 = &frameKeys[y];

		float d00 = norm(frameKey00->pos - frameKey01->pos);
		float d01 = norm(mapKey00->pos - mapKey01->pos);

		float4 d10 = make_float4(frameKey00->pos - frameKey01->pos) / d00;
		float4 d11 = make_float4(mapKey00->pos - mapKey01->pos) / d01;

		if(d00 <= 1e-3 || d01 <= 1e-3)
			score = 0;

		float alpha00 = acos(clamp(frameKey00->normal * frameKey01->normal));
		float beta00 = acos(clamp(d10 * frameKey00->normal));
		float gamma00 = acos(clamp(d10 * frameKey01->normal));

		float alpha01 = acos(clamp(mapKey00->normal * mapKey01->normal));
		float beta01 = acos(clamp(d11 * mapKey00->normal));
		float gamma01 = acos(clamp(d11 * mapKey01->normal));

		score = exp(-(fabs(d00 - d01) + fabs(alpha00 - alpha01) + fabs(beta00 - beta01) + fabs(gamma00 - gamma01)));
	}

	if(isnan(score))
		score = 0;

	adjecencyMatrix.ptr(y)[x] = score;
}

void BuildAdjecencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & frameKeys, DeviceArray<SURF> & mapKeys,
		DeviceArray<float> & dist) {

	int cols = adjecencyMatrix.cols;
	int rows = adjecencyMatrix.rows;

	dim3 thread(8, 8);
	dim3 block(DivUp(cols, thread.x), DivUp(rows, thread.y));

	BuildAdjecencyMatrixKernel<<<block, thread>>>(adjecencyMatrix, frameKeys, mapKeys, dist);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::cuda::GpuMat result;
	cv::cuda::reduce(adjecencyMatrix, result, 0, CV_REDUCE_SUM);
}

__global__ void FilterKeyMatchingKernel(PtrSz<SURF> trainKeys,
		PtrSz<SURF> queryKeys, PtrSz<SURF> trainKeysFiltered,
		PtrSz<SURF> queryKeysFiltered, PtrSz<int> matchesFiltered,
		PtrSz<int> queryIdx, PtrSz<int> keyIdxFiltered) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < matchesFiltered.size) {

		int idx = matchesFiltered[x];
		SURF * trainKey = &trainKeys[idx];
		SURF * queryKey = &queryKeys[idx];

		memcpy((void*) &trainKeysFiltered[x], (void*) trainKey, sizeof(SURF));
		memcpy((void*) &queryKeysFiltered[x], (void*) queryKey, sizeof(SURF));

		keyIdxFiltered[x] = queryIdx[idx];
	}
}

void FilterKeyMatching(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & trainKey, DeviceArray<SURF> & queryKey,
		DeviceArray<SURF> & trainKeyFiltered,
		DeviceArray<SURF> & queryKeyFiltered, DeviceArray<int> & QueryIdx,
		DeviceArray<int> & keyIdxFiltered) {

	cv::cuda::GpuMat result;
	cv::cuda::reduce(adjecencyMatrix, result, 0, CV_REDUCE_SUM);
	cv::Mat cpuResult, indexMat;
	result.download(cpuResult);

	cv::sortIdx(cpuResult, indexMat, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	int selection = indexMat.cols >= 100 ? 100 : indexMat.cols;

	DeviceArray<int> matchFiltered(selection);
	matchFiltered.upload((void*) indexMat.data, selection);

	trainKeyFiltered.create(selection);
	queryKeyFiltered.create(selection);
	keyIdxFiltered.create(selection);

	dim3 thread(MaxThread);
	dim3 block(DivUp(selection, thread.x));

	FilterKeyMatchingKernel<<<block, thread>>>(trainKey, queryKey,
			trainKeyFiltered, queryKeyFiltered, matchFiltered, QueryIdx,
			keyIdxFiltered);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

#include "DeviceMap.h"

#include <opencv.hpp>
#include <cudaarithm.hpp>

__device__ __forceinline__ float clamp(float a) {
	a = a > -1.f ? a : -1.f;
	a = a < 1.f ? a : 1.f;
	return a;
}

__global__ void BuildAdjecencyMatrixKernel(cv::cuda::PtrStepSz<float> adjecencyMatrix,
										   PtrSz<SURF> frameKeys,
										   PtrSz<SURF> mapKeys,
										   PtrSz<float> dist) {

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
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

		float d00 = norm(frameKey00->pos - mapKey00->pos);
		float d01 = norm(frameKey01->pos - mapKey01->pos);

		float4 d10 = make_float4(frameKey00->pos - mapKey00->pos) / d00;
		float4 d11 = make_float4(frameKey01->pos - mapKey01->pos) / d01;

		if(d00 <= 1e-2 || d01 <= 1e-2)
			score = 0;

		float alpha00 = acos(clamp(frameKey00->normal * mapKey00->normal));
		float beta00 = acos(clamp(d10 * frameKey00->normal));
		float gamma00 = acos(clamp(d10 * mapKey00->normal));

		float alpha01 = acos(clamp(frameKey01->normal * mapKey01->normal));
		float beta01 = acos(clamp(d11 * frameKey01->normal));
		float gamma01 = acos(clamp(d11 * mapKey01->normal));

		score = exp(-(fabs(d00 - d01) + fabs(alpha00 - alpha01) + fabs(beta00 - beta01) + fabs(gamma00 - gamma01)));
	}

	if(isnan(score))
		score = 0;

	adjecencyMatrix.ptr(y)[x] = score;
}

void BuildAdjecencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
						  DeviceArray<SURF> & frameKeys,
						  DeviceArray<SURF> & mapKeys,
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
									    PtrSz<SURF> queryKeys,
									    PtrSz<SURF> trainKeysFiltered,
									    PtrSz<SURF> queryKeysFiltered,
									    PtrSz<int> matchesFiltered,
									    PtrSz<int> queryIdx,
									    PtrSz<int> keyIdxFiltered) {

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
					   DeviceArray<SURF> & mapKey,
					   DeviceArray<SURF> & frameKey,
					   DeviceArray<SURF> & mapKeySelected,
					   DeviceArray<SURF> & frameKeySelected,
					   DeviceArray<int> & frameKeyIdx,
					   DeviceArray<int> & keyIdxSelected) {

	cv::cuda::GpuMat result;
	cv::cuda::reduce(adjecencyMatrix, result, 0, CV_REDUCE_SUM);
	cv::Mat cpuResult, index;
	result.download(cpuResult);

	cv::sortIdx(cpuResult, index, CV_SORT_DESCENDING);
	int selection = index.cols >= 100 ? 100 : index.cols;

//	cv::Mat am_cpu(adjecencyMatrix);
//	std::vector<cv::Mat> vmSelectedIdx;
//	cv::Mat cvNoSelected;
//
//	for (int i = 0; i < 1; ++i) {
//
//		cv::Mat mSelectedIdx;
//		int headIdx = 0;
//		int nSelected = 0;
//
//		for (int j = i; j < index.cols; ++j) {
//
//			int idx = index.at<int>(j);
//			if (nSelected == 0) {
//				mSelectedIdx.push_back(idx);
//				headIdx = idx;
//				nSelected++;
//			} else {
//				float score = am_cpu.at<float>(headIdx, idx);
//				if (score > 0.1f) {
//					mSelectedIdx.push_back(idx);
//					nSelected++;
//				}
//			}
//
//			if (nSelected >= 100)
//				break;
//		}
//
//		if (nSelected >= 4) {
//			cvNoSelected.push_back(nSelected);
//			vmSelectedIdx.push_back(mSelectedIdx);
//		}
//	}
//	cv::Mat tmp;
//	cv::sortIdx(cvNoSelected, tmp, CV_SORT_DESCENDING);
//	index = vmSelectedIdx[0];
//	selection = index.cols;

	DeviceArray<int> MatchIdxSelected(selection);
	MatchIdxSelected.upload((void*) index.data, selection);

	mapKeySelected.create(selection);
	frameKeySelected.create(selection);
	keyIdxSelected.create(selection);

	dim3 thread(MaxThread);
	dim3 block(DivUp(selection, thread.x));

	FilterKeyMatchingKernel<<<block, thread>>>(mapKey, frameKey, mapKeySelected,
			frameKeySelected, MatchIdxSelected, frameKeyIdx, keyIdxSelected);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

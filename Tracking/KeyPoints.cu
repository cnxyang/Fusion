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

		float d00 = norm(frameKey00->pos - frameKey01->pos);
		float d01 = norm(mapKey00->pos - mapKey01->pos);

		float4 d10 = make_float4(normalised(frameKey00->pos - frameKey01->pos));
		float4 d11 = make_float4(normalised(mapKey00->pos - mapKey01->pos));

		if(d00 <= 1e-2 || d01 <= 1e-2) {
			score = 0;
		} else {
			float alpha00 = acos(clamp(frameKey00->normal * frameKey01->normal));
			float beta00 = acos(clamp(d10 * frameKey00->normal));
			float gamma00 = acos(clamp(d10 * frameKey01->normal));
			float alpha01 = acos(clamp(mapKey00->normal * mapKey01->normal));
			float beta01 = acos(clamp(d11 * mapKey00->normal));
			float gamma01 = acos(clamp(d11 * mapKey01->normal));
			score = exp(-(fabs(d00 - d01) + fabs(alpha00 - alpha01) + fabs(beta00 - beta01) + fabs(gamma00 - gamma01)));
		}
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

	if(cpuResult.cols == 0)
		return;

	cv::sortIdx(cpuResult, index, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

	cv::Mat am_cpu(adjecencyMatrix);
	std::vector<cv::Mat> vmSelectedIdx;
	cv::Mat cvNoSelected;

	for (int i = 0; i < 10; ++i) {

		cv::Mat mSelectedIdx;
		int headIdx = 0;
		int nSelected = 0;

		for (int j = i; j < index.cols; ++j) {

			int idx = index.at<int>(j);
			if (nSelected == 0) {
				mSelectedIdx.push_back(idx);
				headIdx = idx;
				nSelected++;
			} else {
				float score = am_cpu.at<float>(headIdx, idx);
				if (score > 0.1f) {
					mSelectedIdx.push_back(idx);
					nSelected++;
				}
			}

			if (nSelected >= 400)
				break;
		}

		if (nSelected >= 3) {
			cv::Mat refined;
			for (int k = 1; k < nSelected; ++k) {
				int a = mSelectedIdx.at<int>(k);
				int l = k + 1;
				for (; l < nSelected; ++l) {
					int b = mSelectedIdx.at<int>(l);
					if(k != l && (am_cpu.at<float>(a, b) < 0.005f || am_cpu.at<float>(b, a) < 0.005f)) {
						if(am_cpu.at<float>(headIdx, b) > am_cpu.at<float>(headIdx, a))
							break;
					}
				}
				if(l >= nSelected) {
					refined.push_back(a);
				}
			}

			cvNoSelected.push_back(refined.rows);
			vmSelectedIdx.push_back(refined.t());
		}
	}

	cv::Mat tmp;
	if(cvNoSelected.rows == 0)
		return;

	cv::sortIdx(cvNoSelected, tmp, CV_SORT_DESCENDING);
	index = vmSelectedIdx[tmp.at<int>(0)];
	int selection = index.cols;

	if(selection <= 3)
		return;

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

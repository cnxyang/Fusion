#include "DeviceMap.h"

#include <opencv.hpp>
#include <cudaarithm.hpp>

__global__ void BuildAdjecencyMatrixKernel(
		cv::cuda::PtrStepSz<float> adjecencyMatrix, PtrSz<SurfKey> frameKeys,
		PtrSz<SurfKey> mapKeys, PtrSz<float> dist) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= adjecencyMatrix.cols || y >= adjecencyMatrix.rows)
		return;

	float score = 0;
	if(x == y)
		score = exp(-dist[x]);
	else {
		SurfKey * frameKey00 = &frameKeys[x];
		SurfKey * mapKey00 = &mapKeys[x];
		SurfKey * frameKey01 = &frameKeys[y];
		SurfKey * mapKey01 = &mapKeys[y];
		float d00 = norm(frameKey00->pos - frameKey01->pos);
		float d01 = norm(mapKey00->pos - mapKey01->pos);
		if(d00 > 1e-3 && d01 > 1e-3) {
			float alpha00 = acos(frameKey00->normal * frameKey01->normal);
			float alpha01 = acos(mapKey00->normal * mapKey01->normal);
			score = exp(-(fabs(d00 - d01) + fabs(alpha00 - alpha01)));
		}
	}

	if(isnan(score))
		score = 0;
	adjecencyMatrix.ptr(y)[x] = score;
}

void BuildAdjecencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SurfKey> & frameKeys, DeviceArray<SurfKey> & mapKeys,
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

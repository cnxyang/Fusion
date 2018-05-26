#include "DeviceFunc.h"

template <class T, class U, int size> __global__
void BilateralFiltering_device(const PtrStepSz<T> src, PtrStep<U> dst, float s, float r, float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= src.cols || y >= src.rows)
    	return;

    int minX = max(0, x - size / 2);
    int maxX = min(x + size / 2 + 1, src.cols);
    int minY = max(0, y - size / 2);
    int maxY = min(y + size / 2 + 1, src.rows);

    float val = 0, weight = 0;
    float valc = src.ptr(y)[x] * scale;
    for(int i = minX; i < maxX; ++i) {
    	for(int j = minY; j < maxY; ++j) {
    		float valp = src.ptr(j)[i] * scale;
    		float gs2 = (x - i) * (x - i) + (y - j) * (y - j);
    		float gr2 = (valc - valp) * (valc - valp);
    		float wp = __expf(-gs2 * s - gr2 * r);
    		val += wp * valp;
    		weight += wp;
    	}
    }
    if(weight < 1e-6)
    	dst.ptr(y)[x] = (U)valc;
    else
    	dst.ptr(y)[x] = (U)(val / weight);
}

void BilateralFiltering(const DeviceArray2D<ushort>& src, DeviceArray2D<float>& dst, float scale) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	float SigmaSpace = 0.5 / (4 * 4);
	float SigmaRange = 0.5 / (0.5 * 0.5);
	BilateralFiltering_device<ushort, float, 5><<<grid, block>>>(src, dst, SigmaSpace, SigmaRange, 1.0 / scale);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

template <class T, class U> __global__
void PyrDownGaussian_device(const PtrStepSz<T> src, PtrStep<U> dst, float* kernel) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= dst.cols || y >= dst.rows)
		return;

	const int D = 5;
	float center = src.ptr(2 * y)[2 * x];
	int tx = min(2 * x - D / 2 + D, src.cols - 1);
	int ty = min(2 * y - D / 2 + D, src.rows - 1);
	int cy = max(0, 2 * y - D / 2);
	float sum = 0;
	int count = 0;
	for (; cy < ty; ++cy) {
		for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
			if (!isnan((float)src.ptr(cy)[cx])) {
				sum += src.ptr(cy)[cx] * kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
				count += kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
			}
		}
	}

	dst.ptr(y)[x] = (float) (sum / (float) count);
}

void PyrDownGaussian(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(dst.cols(), block.x), cv::divUp(dst.rows(), block.y));

    const float gaussKernel[25] = {1, 4,  6,  4,  1,
    							   4, 16, 24, 16, 4,
    							   6, 24, 36, 24, 6,
    							   4, 16, 24, 16, 4,
    							   1, 4,  6,  4,  1};

    DeviceArray<float> kernel(25);
    kernel.upload((void*)gaussKernel, 25);

	PyrDownGaussian_device<float, float><<<grid, block>>>(src, dst, kernel);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void PyrDownGaussian(const DeviceArray2D<uchar>& src, DeviceArray2D<uchar>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

    const float gaussKernel[25] = {1, 4,  6,  4,  1,
    							   4, 16, 24, 16, 4,
    							   6, 24, 36, 24, 6,
    							   4, 16, 24, 16, 4,
    							   1, 4,  6,  4,  1};

    DeviceArray<float> kernel(25);
    kernel.upload((void*)gaussKernel, 25);

	PyrDownGaussian_device<uchar, uchar><<<grid, block>>>(src, dst, kernel);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ComputeDerivativeImage_device(PtrStepSz<uchar> src, PtrStep<short> dIx, PtrStep<short> dIy) {

}

void ComputeDerivativeImage(const DeviceArray2D<uchar>& src, DeviceArray2D<short>& dIx, DeviceArray2D<short>& dIy) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ComputeDerivativeImage_device<<<grid, block>>>(src, dIx, dIy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ColourImageToIntensity_device(PtrStepSz<uchar3> src, PtrStep<uchar> dst) {
	 int x = blockIdx.x * blockDim.x + threadIdx.x;
	 int y = blockIdx.y * blockDim.y + threadIdx.y;
	 if (x >= dst.cols || y >= dst.rows)
		 return;

	 uchar3 val = src.ptr(y)[x];
	 int value = (float)val.x * 0.2126f + (float)val.y * 0.7152f + (float)val.z * 0.0722f;
	 dst.ptr (y)[x] = value;
}

void ColourImageToIntensity(const DeviceArray2D<uchar3>& src, DeviceArray2D<uchar>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ColourImageToIntensity_device<<<grid, block>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

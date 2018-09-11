#include "cufunc.h"

template<class T, class U, int size> __global__
void BilateralFiltering_device(const PtrStepSz<T> src, PtrStep<U> dst, float s,
		float r, float scale) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	int minX = max(0, x - size / 2);
	int maxX = min(x + size / 2 + 1, src.cols);
	int minY = max(0, y - size / 2);
	int maxY = min(y + size / 2 + 1, src.rows);

	float val = 0, weight = 0;
	float valc = src.ptr(y)[x] * scale;
	for (int i = minX; i < maxX; ++i) {
		for (int j = minY; j < maxY; ++j) {
			float valp = src.ptr(j)[i] * scale;
			float gs2 = (x - i) * (x - i) + (y - j) * (y - j);
			float gr2 = (valc - valp) * (valc - valp);
			float wp = __expf(-gs2 * s - gr2 * r);
			val += wp * valp;
			weight += wp;
		}
	}
	if (weight < 1e-6)
		dst.ptr(y)[x] = (U) valc;
	else
		dst.ptr(y)[x] = (U) (val / weight);
}

void BilateralFiltering(const DeviceArray2D<ushort>& src,
		DeviceArray2D<float>& dst, float scale) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	float SigmaSpace = 0.5 / (4 * 4);
	float SigmaRange = 0.5 / (0.5 * 0.5);
	BilateralFiltering_device<ushort, float, 5> <<<grid, block>>>(src, dst,
			SigmaSpace, SigmaRange, 1.0 / scale);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

template<class T, class U> __global__
void PyrDownGaussian_device(const PtrStepSz<T> src, PtrStepSz<U> dst,
		float* kernel) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dst.cols || y >= dst.rows)
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
			if (!isnan((float) src.ptr(cy)[cx])) {
				sum += src.ptr(cy)[cx]
						* kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
				count += kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
			}
		}
	}

	dst.ptr(y)[x] = (U) (sum / (float) count);
}

void PyrDownGaussian(const DeviceArray2D<float>& src,
		DeviceArray2D<float>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(dst.cols(), block.x), cv::divUp(dst.rows(), block.y));

	const float gaussKernel[25] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36,
			24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1 };

	DeviceArray<float> kernel(25);
	kernel.upload((void*) gaussKernel, 25);

	PyrDownGaussian_device<float, float> <<<grid, block>>>(src, dst, kernel);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void PyrDownGaussian(const DeviceArray2D<uchar>& src,
		DeviceArray2D<uchar>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	const float gaussKernel[25] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36,
			24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1 };

	DeviceArray<float> kernel(25);
	kernel.upload((void*) gaussKernel, 25);

	PyrDownGaussian_device<uchar, uchar> <<<grid, block>>>(src, dst, kernel);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ColourImageToIntensity_device(PtrStepSz<uchar3> src,
		PtrStep<uchar> dst) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	uchar3 val = src.ptr(y)[x];
	int value = (float) val.x * 0.2126f + (float) val.y * 0.7152f
			+ (float) val.z * 0.0722f;
	dst.ptr(y)[x] = value;
}

void ColourImageToIntensity(const DeviceArray2D<uchar3>& src,
		DeviceArray2D<uchar>& dst) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ColourImageToIntensity_device<<<grid, block>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__constant__ int sobely[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
__constant__ int sobelx[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };

__global__ void ComputeDerivativeImage_device(PtrStepSz<uchar> src,
		PtrStep<float> dIx, PtrStep<float> dIy) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	if (x > 0 && y > 0 && x < src.cols - 1 && y < src.rows - 1) {

		int dx = 0;
		int dy = 0;
		int id = 8;
		for (int i = -1; i < 2; ++i)
			for (int j = -1; j < 2; ++j) {
				int val = src.ptr(y + i)[x + j];
				dx += val * sobelx[id];
				dy += val * sobely[id];
				--id;
			}
		dIx.ptr(y)[x] = (float) dx / 8;
		dIy.ptr(y)[x] = (float) dy / 8;
	} else {
		dIx.ptr(y)[x] = 0;
		dIy.ptr(y)[x] = 0;
	}
}

void ComputeDerivativeImage(const DeviceArray2D<uchar>& src,
		DeviceArray2D<float>& dx, DeviceArray2D<float>& dy) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ComputeDerivativeImage_device<<<grid, block>>>(src, dx, dy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResizeMap_device(const PtrStepSz<float4> vsrc,
		const PtrStep<float4> nsrc, PtrStepSz<float4> vdst,
		PtrStep<float4> ndst) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= vsrc.cols || y >= vsrc.rows)
		return;

	float4 v00 = vsrc.ptr(y * 2 + 0)[x * 2 + 0];
	float4 v01 = vsrc.ptr(y * 2 + 0)[x * 2 + 1];
	float4 v10 = vsrc.ptr(y * 2 + 1)[x * 2 + 0];
	float4 v11 = vsrc.ptr(y * 2 + 1)[x * 2 + 1];
	float4 n00 = nsrc.ptr(y * 2 + 0)[x * 2 + 0];
	float4 n01 = nsrc.ptr(y * 2 + 0)[x * 2 + 1];
	float4 n10 = nsrc.ptr(y * 2 + 1)[x * 2 + 0];
	float4 n11 = nsrc.ptr(y * 2 + 1)[x * 2 + 1];

	if (isnan(v00.x) || isnan(v01.x) || isnan(v10.x) || isnan(v11.x)) {
		vdst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	} else {
		vdst.ptr(y)[x] = (v00 + v01 + v10 + v11) / 4;
	}

	if (isnan(n00.x) || isnan(n01.x) || isnan(n10.x) || isnan(n11.x)) {
		ndst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	} else {
		ndst.ptr(y)[x] = normalised((n00 + n01 + n10 + n11) / 4);
	}
}

void ResizeMap(const DeviceArray2D<float4>& vsrc,
		const DeviceArray2D<float4>& nsrc, DeviceArray2D<float4>& vdst,
		DeviceArray2D<float4>& ndst) {

//	vdst.create(vsrc.cols() / 2, vsrc.rows() / 2);
//	ndst.create(nsrc.cols() / 2, nsrc.rows() / 2);

	dim3 block(8, 8);
	dim3 grid(cv::divUp(vdst.cols(), block.x), cv::divUp(vdst.rows(), block.y));

	ResizeMap_device<<<grid, block>>>(vsrc, nsrc, vdst, ndst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

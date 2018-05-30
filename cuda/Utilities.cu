#include "Converter.h"
#include "DeviceFunc.h"
#include "DeviceMath.h"
#include "DeviceArray.h"

__global__ void
BackProjectPoints_device(const PtrStepSz<float> src,
										   PtrStepSz<float4> dst,
										   float depthCutoff, float invfx, float invfy,
										   float cx, float cy) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	float4 point;
	point.z = src.ptr(y)[x];
	if(!isnan(point.z) && point.z > 1e-3) {
		point.x = point.z * (x - cx) * invfx;
		point.y = point.z * (y - cy) * invfy;
		point.w = 1.0;
	}
	else
		point.x = __int_as_float(0x7fffffff);

	dst.ptr(y)[x] = point;
}

void BackProjectPoints(const DeviceArray2D<float>& src,
									   DeviceArray2D<float4>& dst, float depthCutoff,
									   float fx, float fy, float cx, float cy) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	BackProjectPoints_device<<<grid, block>>>(src, dst, depthCutoff, 1.0 / fx, 1.0 / fy, cx, cy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ComputeNormalMap_device(const PtrStepSz<float4> src, PtrStepSz<float3> dst) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	if(x == src.cols - 1 || y == src.rows - 1) {
		dst.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));
		return;
	}

	float4 vcentre = src.ptr(y)[x];
	float4 vright = src.ptr(y)[x + 1];
	float4 vdown = src.ptr(y + 1)[x];

	if(!isnan(vcentre.x) && !isnan(vright.x) && !isnan(vdown.x)) {
		dst.ptr(y)[x] = normalised(cross(vright - vcentre, vdown - vcentre));
	}
	else
		dst.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));
}

void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float3>& dst) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ComputeNormalMap_device<<<grid, block>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
WarpGrayScaleImage_device(PtrStepSz<float4> src, PtrStep<uchar> gray,
						  	  	  	  	  	     Matrix3f R1, Matrix3f invR2, float3 t1, float3 t2,
						  	  	  	  	  	     float fx, float fy, float cx, float cy,
						  	  	  	  	  	     PtrStep<uchar> diff) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	diff.ptr(y)[x] = 0;

	float3 srcp = make_float3(src.ptr(y)[x]);
	if(isnan(srcp.x) || srcp.z < 1e-6)
		return;
	float3 dst = R1 * srcp + t1;
	dst = invR2 * (dst - t2);

	int u = __float2int_rd(fx * dst.x / dst.z + cx + 0.5);
	int v = __float2int_rd(fy * dst.y / dst.z + cy + 0.5);
	if(u >= 0 && v >= 0 && u < src.cols && v < src.rows)
		diff.ptr(y)[x] = gray.ptr(v)[u];
}

void WarpGrayScaleImage(const Frame& frame1, const Frame& frame2,
										    DeviceArray2D<uchar>& diff) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(diff.cols(), block.x), cv::divUp(diff.rows(), block.y));

	float3 t1 = Converter::CvMatToFloat3(frame1.mtcw);
	float3 t2 = Converter::CvMatToFloat3(frame2.mtcw);

	const int pyrlvl = 0;

	WarpGrayScaleImage_device<<<grid, block>>>(frame1.mVMap[pyrlvl], frame2.mGray[pyrlvl],
																				    frame1.mRcw, frame2.mRwc, t1, t2,
																				    Frame::fx(pyrlvl), Frame::fy(pyrlvl),
																				    Frame::cx(pyrlvl), Frame::cy(pyrlvl), diff);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ComputeResidualImage_device(PtrStepSz<uchar> src,
												     PtrStep<uchar> dst,
												     PtrStep<uchar> residual) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	residual.ptr(y)[x] = abs(src.ptr(y)[x] - dst.ptr(y)[x]);
}

void ComputeResidualImage(const DeviceArray2D<uchar>& src,
										        DeviceArray2D<uchar>& residual,
										        const Frame& frame) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(residual.cols(), block.x), cv::divUp(residual.rows(), block.y));

	const int pyrlvl = 0;

	ComputeResidualImage_device<<<grid, block>>>(src, frame.mGray[pyrlvl], residual);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

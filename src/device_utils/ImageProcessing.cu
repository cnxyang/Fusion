#include "DeviceFuncs.h"

__global__ void scale_depth_kernel(const PtrStepSz<unsigned short> raw_depth, PtrStep<float> depth, float scale_inv)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= raw_depth.cols || y >= raw_depth.rows)
		return;

    float z = raw_depth.ptr(y)[x] * scale_inv;
    depth.ptr(y)[x] = (z == z ? z : 0);
}

void FilterDepth(const DeviceArray2D<unsigned short> & depth,
		DeviceArray2D<float> & rawDepth, DeviceArray2D<float> & filteredDepth,
		float depthScale, float depthCutoff) {

	dim3 thread(8, 8);
	dim3 block(divUp(depth.cols, thread.x), divUp(depth.rows, thread.y));

	scale_depth_kernel<<<block, thread>>>(depth, rawDepth, 1.0 / depthScale);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void compute_vertex_map_kernel(const PtrStepSz<float> depth, PtrStep<float4> vmap, float invfx, float invfy,	float cx, float cy) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= depth.cols || y >= depth.rows)
		return;

	float4 v;
	v.z = depth.ptr(y)[x];
	if(v.z > 0.2)
	{
		v.x = v.z * (x - cx) * invfx;
		v.y = v.z * (y - cy) * invfy;
		v.w = 1.0;
	}
	else
	{
		v.x = __int_as_float(0x7fffffff);
	}

	vmap.ptr(y)[x] = v;
}

void ComputeVMap(const DeviceArray2D<float> & depth, DeviceArray2D<float4> & vmap, float fx, float fy, float cx, float cy, float depthCutoff) {

	dim3 thread(8, 8);
	dim3 block(divUp(depth.cols, thread.x), divUp(depth.rows, thread.y));

	compute_vertex_map_kernel<<<block, thread>>>(depth, vmap, 1.0 / fx, 1.0 / fy, cx, cy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());


}

__global__ void compute_normal_map_kernel(const PtrStepSz<float4> vmap, PtrStep<float4> nmap)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= vmap.cols || y >= vmap.rows)
		return;
	nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	if (x < 1 || y < 1 || x >= vmap.cols - 1 || y >= vmap.rows - 1)
		return;

	float4 left = vmap.ptr(y)[x - 1];
	float4 right = vmap.ptr(y)[x + 1];
	float4 up = vmap.ptr(y + 1)[x];
	float4 down = vmap.ptr(y - 1)[x];

	if(left == left && right == right && up == up && down == down)
	{
		nmap.ptr(y)[x] = make_float4(normalised(cross(left - right , up - down)), 1.f);
	}
}

void ComputeNMap(const DeviceArray2D<float4> & vmap, DeviceArray2D<float4> & nmap)
{
	dim3 block(8, 8);
	dim3 grid(divUp(vmap.cols, block.x), divUp(vmap.rows, block.y));

	compute_normal_map_kernel<<<grid, block>>>(vmap, nmap);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void pyrdown_mean_smooth(const PtrStep<float> src, PtrStepSz<float> dst)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dst.cols || y >= dst.rows)
		return;

	int x0 = x * 2;
	int x1 = x0 + 1;
	int y0 = y * 2;
	int y1 = y0 + 1;

	dst.ptr(y)[x] = (src.ptr(y0)[x0] + src.ptr(y0)[x1] + src.ptr(y1)[x0] + src.ptr(y1)[x1]) / 4.0f;
}

__global__ void pyrdown_subsample(const PtrStep<float> src, PtrStepSz<float> dst)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dst.cols || y >= dst.rows)
		return;

	dst.ptr(y)[x] = src.ptr(y * 2)[x * 2];
}

void pyrdown_image_mean(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst)
{
	dim3 thread(8, 8);
	dim3 block(divUp(dst.cols, thread.x), divUp(dst.rows, thread.y));

	pyrdown_subsample<<<block, thread>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst) {

	dim3 thread(8, 8);
	dim3 block(divUp(dst.cols, thread.x), divUp(dst.rows, thread.y));

	pyrdown_subsample<<<block, thread>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void compute_image_derivatives_kernel(const PtrStepSz<float> image, PtrStep<float> dx, PtrStep<float> dy)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= image.cols || y >= image.rows)
		return;

	float dxVal = 0;
	float dyVal = 0;

	if(x < 1 || y < 1 || x >= image.cols - 1 || y >= image.rows - 1)
	{

	}
	else
	{
		dxVal = (image.ptr(y)[x + 1] - image.ptr(y)[x - 1]) / 2.0f;
		dyVal = (image.ptr(y + 1)[x] - image.ptr(y - 1)[x]) / 2.0f;
	}

	dx.ptr(y)[x] = dxVal;
	dy.ptr(y)[x] = dyVal;
}

void compute_image_derivatives(const DeviceArray2D<float>& image, DeviceArray2D<float>& dx, DeviceArray2D<float>& dy)
{
    dim3 block(8, 8);
    dim3 grid(divUp(image.cols, block.x), divUp(image.rows, block.y));

    compute_image_derivatives_kernel<<<grid, block>>>(image, dx, dy);

    SafeCall(cudaDeviceSynchronize());
    SafeCall(cudaGetLastError());
}

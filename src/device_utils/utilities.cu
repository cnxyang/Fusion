#include "DeviceFuncs.h"

__global__ void compute_residual_image_kernel(const PtrStep<uchar> image_curr, const PtrStep<uchar> image_last, PtrStepSz<uchar> residual_image)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= residual_image.cols || y >= residual_image.rows)
		return;

	residual_image.ptr(y)[x] = abs((float)image_curr.ptr(y)[x] - image_last.ptr(y)[x]);
}

void compute_residual_image(const DeviceArray2D<uchar>& image_curr, const DeviceArray2D<uchar>& image_last)
{
	DeviceArray2D<uchar> residual_image(image_curr.cols, image_curr.rows);
	dim3 block(32, 8);
	dim3 thread(divUp(image_curr.cols, block.x), divUp(image_curr.rows, block.y));

	compute_residual_image_kernel<<<thread, block>>>(image_curr, image_last, residual_image);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat residual_img(image_curr.rows, image_curr.cols, CV_8UC1);
	residual_image.download((void*)residual_img.data, residual_img.step);
	cv::imshow("residual image", residual_img);
	cv::waitKey(0);
	cv::imwrite("1.jpg", residual_img);
}

__global__ void compute_residual_transformed_kernel(
		const PtrStepSz<float4> vmap_curr,
		const PtrStep<float4> vmap_last,
		const PtrStep<float4> nmap_last,
		const PtrStep<float> image_curr,
		const PtrStep<float> image_last,
		PtrStep<float> residual_image,
		PtrStep<float> residual_icp,
		float fx, float fy, float cx, float cy,
		Matrix3f r, float3 t)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= vmap_curr.cols || y >= vmap_curr.rows)
		return;
	residual_icp.ptr(y)[x] = 0;
	residual_image.ptr(y)[x] = 0;
	float3 v_g = r * make_float3(vmap_curr.ptr(y)[x]) + t;
	int u = __float2int_rd(fx * v_g.x / v_g.z + cx + 0.5);
	int v = __float2int_rd(fy * v_g.y / v_g.z + cy + 0.5);
	if(u < 0 || v < 0 || u >= vmap_curr.cols || v >= vmap_curr.rows)
	{
		return;
	}

	float i_c = image_curr.ptr(y)[x];
	float i_l = image_last.ptr(v)[u];
	if(i_c == 0 || i_l == 0)
	{
		return;
	}

	float3 v_l = make_float3(vmap_last.ptr(v)[u]);
	float3 n_l = make_float3(nmap_last.ptr(v)[u]);

	if(v_g == v_g && v_l == v_l && n_l == n_l)
	{
		residual_icp.ptr(y)[x] = n_l * (v_g - v_l) * 10000;
		residual_image.ptr(y)[x] = abs(i_l - i_c);
	}
}

void compute_residual_transformed(const DeviceArray2D<float4>& vmap_curr, const DeviceArray2D<float4>& vmap_last, const DeviceArray2D<float4>& nmap_last, const DeviceArray2D<float>& image_curr, const DeviceArray2D<float>& image_last,	float* K, Matrix3f r, float3 t)
{
	int cols = vmap_curr.cols;
	int rows = vmap_curr.rows;

	DeviceArray2D<float> r_icp(cols, rows);
	DeviceArray2D<float> r_img(cols, rows);

	dim3 block(8, 8);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

	compute_residual_transformed_kernel<<<grid, block>>>(vmap_curr, vmap_last, nmap_last, image_curr, image_last, r_img, r_icp, K[0], K[1], K[2], K[3], r, t);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat residual_img(rows, cols, CV_32FC1);
	r_img.download((void*)residual_img.data, residual_img.step);
	cv::Mat residual_icp(rows, cols, CV_32FC1);
	r_icp.download((void*)residual_icp.data, residual_icp.step);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());


	residual_icp.convertTo(residual_icp, CV_8UC1);
	residual_img.convertTo(residual_img, CV_8UC1);
	cv::imshow("residual image", residual_img);
	cv::imshow("residual icp", residual_icp);
	cv::waitKey(0);
}

void compute_residual_transformed_gt(const DeviceArray2D<float4>& vmap_curr, const DeviceArray2D<float4>& vmap_last, const DeviceArray2D<float4>& nmap_last, const DeviceArray2D<float>& image_curr, const DeviceArray2D<float>& image_last, float* K, Matrix3f r_gt, float3 t_gt, Matrix3f r, float3 t)
{
	int cols = vmap_curr.cols;
	int rows = vmap_curr.rows;

	DeviceArray2D<float> r_icp(cols, rows);
	DeviceArray2D<float> r_img(cols, rows);
	DeviceArray2D<float> r_icp_gt(cols, rows);
	DeviceArray2D<float> r_img_gt(cols, rows);

	dim3 block(8, 8);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

	compute_residual_transformed_kernel<<<grid, block>>>(vmap_curr, vmap_last, nmap_last, image_curr, image_last, r_img, r_icp, K[0], K[1], K[2], K[3], r, t);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat residual_img(rows, cols, CV_32FC1);
	r_img.download((void*)residual_img.data, residual_img.step);
	cv::Mat residual_icp(rows, cols, CV_32FC1);
	r_icp.download((void*)residual_icp.data, residual_icp.step);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());


	residual_icp.convertTo(residual_icp, CV_8UC1);
	residual_img.convertTo(residual_img, CV_8UC1);
	cv::imshow("residual image", residual_img);
	cv::imshow("residual icp", residual_icp);

	compute_residual_transformed_kernel<<<grid, block>>>(vmap_curr, vmap_last, nmap_last, image_curr, image_last, r_img_gt, r_icp_gt, K[0], K[1], K[2], K[3], r_gt, t_gt);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat residual_img_gt(rows, cols, CV_32FC1);
	r_img_gt.download((void*)residual_img_gt.data, residual_img_gt.step);
	cv::Mat residual_icp_gt(rows, cols, CV_32FC1);
	r_icp_gt.download((void*)residual_icp_gt.data, residual_icp_gt.step);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	residual_icp_gt.convertTo(residual_icp_gt, CV_8UC1);
	residual_img_gt.convertTo(residual_img_gt, CV_8UC1);
	cv::imshow("residual image gt", residual_img_gt);
	cv::imshow("residual icp gt", residual_icp_gt);
	cv::waitKey(0);
}


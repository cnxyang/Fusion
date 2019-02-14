#include "DeviceFuncs.h"
#include <Eigen/Dense>
#define WarpSize 32
#define MaxThread 1024

template<int rows, int cols> void inline CreateMatrix(float* host_data, double* host_a, double* host_b) {
	int shift = 0;
	for (int i = 0; i < rows; ++i)
		for (int j = i; j < cols; ++j) {
			double value = (double) host_data[shift++];
			if (j == rows)
				host_b[i] = value;
			else
				host_a[j * rows + i] = host_a[i * rows + j] = value;
		}
}

template<typename T, int size> __device__ inline void WarpReduce(T* val) {
	#pragma unroll
	for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
		#pragma unroll
		for (int i = 0; i < size; ++i) {
			val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
		}
	}
}

template<typename T, int size> __device__ inline void BlockReduce(T* val) {
	static __shared__ T shared[32 * size];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	WarpReduce<T, size>(val);

	if (lane == 0)
		memcpy(&shared[wid * size], val, sizeof(T) * size);

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(val, &shared[lane * size], sizeof(T) * size);
	else
		memset(val, 0, sizeof(T) * size);

	if (wid == 0)
		WarpReduce<T, size>(val);
}

template<typename T, int size> __global__ void Reduce(PtrStep<T> in, T * out, int N) {
	T sum[size];
	memset(sum, 0, sizeof(T) * size);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < N; i += blockDim.x * gridDim.x)
	#pragma unroll
		for (int j = 0; j < size; ++j)
			sum[j] += in.ptr(i)[j];

	BlockReduce<T, size>(sum);

	if (threadIdx.x == 0)
	#pragma unroll
		for (int i = 0; i < size; ++i)
			out[i] = sum[i];
}

//class ResidualSum
//{
//public:
//
//	__device__ __forceinline__ bool associate_points(int& x, int& y, int& u, int& v)
//	{
//		float3 vertex_curr = make_float3(vmap_curr_.ptr(y)[x]);
//		vertex_curr_last_ = rcurr_ * vertex_curr + tcurr_;
//		u = __float2int_rd(vertex_curr_last_.x / vertex_curr_last_.z * fx_ + cx_ + 0.5f);
//		v = __float2int_rd(vertex_curr_last_.y / vertex_curr_last_.z * fy_ + cy_ + 0.5f);
//		if(u < 0 || v < 0 || u >= width_ || v >= height_)
//			return false;
//
//		float3 normal_curr = make_float3(nmap_curr_.ptr(y)[x]);
//		float3 normal_curr_last = rcurr_ * normal_curr;
//
//		vertex_last_ = make_float3(vmap_last_.ptr(v)[u]);
//		normal_last_ = make_float3(nmap_last_.ptr(v)[u]);
//
//		float dist = norm(vertex_curr_last_ - vertex_last_);
//		float angle = norm(cross(normal_curr_last, normal_last_));
//
//		printf("%f\n", dist);
//
//		return (dist < dist_thresh_ && angle <= angle_thresh_ && !isnan(normal_curr.x) && !isnan(normal_last_.x));
//	}
//
//	__device__ __forceinline__ bool search_correspondence(int& x, int& y, int& u, int& v)
//	{
//		bool valid = false;
//		if (x >= 5 && x < width_ - 5 && y >= 5 && y < height_ - 5) {
//
//			valid = associate_points(x, y, u, v);
//			for (int j = max(y - 2, 0); j < min(y + 2, height_); ++j) {
//				for (int k = max(x - 2, 0); k < min(x + 2, width_); ++k) {
//					valid = valid && (image_curr_.ptr(j)[k] > 0);
//				}
//			}
//		}
//
//		if(valid)
//		{
//			var_curr_ = image_curr_.ptr(y)[x];
//			var_last_ = image_last_.ptr(v)[u];
////			if(abs(var_curr_ - var_last_) < var_thresh_)
////			if(var_curr_ > 0 && var_last_ > 0)
////			{
//				return true;
////			}
//		}
//
//		return false;
//	}
//
//	__device__ __forceinline__ void compute_residual(int& k, float* value)
//	{
//		int y = k / width_;
//		int x = k - y * width_;
//		int u, v;
//		CorrespItem& item = corresp_image_.ptr(y)[x];
//		item.valid = false;
//		value[0] = 0;
//		value[1] = 0;
//		value[2] = 0;
//		if(search_correspondence(x, y, u, v))
//		{
//			item.valid = true;
//			item.u = u;
//			item.v = v;
//			value[0] = normal_last_ * (vertex_curr_last_ - vertex_last_);
//			value[1] = var_curr_ - var_last_;
//			value[2] = 1.f;
//			item.icp_residual = value[0];
//			item.rgb_residual = value[1];
//		}
//	}
//
//	__device__ __inline__ void operator()()
//	{
//		float sum[3] = { 0, 0, 0 };
//		float value[3];
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		for (; i < width_ * height_; i += blockDim.x * gridDim.x)
//		{
//			compute_residual(i, value);
//#pragma unroll
//			for (int j = 0; j < 3; ++j)
//				sum[j] += value[j];
//		}
//
//		BlockReduce<float, 3>(sum);
//
//		if (threadIdx.x == 0)
//#pragma unroll
//			for (int i = 0; i < 3; ++i)
//				out.ptr(blockIdx.x)[i] = sum[i];
//	}
//
//	PtrStep<float4> vmap_curr_, vmap_last_, nmap_curr_, nmap_last_;
//	PtrStep<unsigned char> image_curr_, image_last_;
//	Matrix3f rcurr_;
//	float3 tcurr_;
//	float fx_, fy_, cx_, cy_;
//	int width_, height_;
//	float angle_thresh_, dist_thresh_;
//	int var_thresh_;
//
//	PtrStep<CorrespItem> corresp_image_;
//	PtrStep<float> out;
//
//private:
//
//	float3 vertex_curr_last_, vertex_last_, normal_last_;
//	unsigned char var_curr_, var_last_;
//	float icp_residual_, rgb_residual_;
//};
//
//__global__ void compute_residual_sum_kernel(ResidualSum rs)
//{
//	rs();
//}
//
//class ComputeWeightMatrix
//{
//public:
//
//	__device__ __forceinline__ void compute_variance_covariance(int& k, float* value)
//	{
//		int y = k / width_;
//		int x = k - y * width_;
//
//#pragma unroll
//		for (int j = 0; j < 3; ++j)
//			value[j] = 0;
//
//		CorrespItem& item = corresp_image_.ptr(y)[x];
//		if(item.valid)
//		{
//			value[0] = (item.icp_residual - mean_icp_) * (item.icp_residual - mean_icp_);
//			value[1] = (item.rgb_residual - mean_rgb_) * (item.rgb_residual - mean_rgb_);
//			value[2] = (item.rgb_residual - mean_rgb_) * (item.icp_residual - mean_icp_);
//		}
//	}
//
//	__device__ __inline__ void operator()()
//	{
//		float sum[3] = { 0, 0, 0 };
//		float value[3];
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		for (; i < N; i += blockDim.x * gridDim.x)
//		{
//			compute_variance_covariance(i, value);
//#pragma unroll
//			for (int j = 0; j < 3; ++j)
//				sum[j] += value[j];
//		}
//
//		BlockReduce<float, 3>(sum);
//
//		if (threadIdx.x == 0)
//#pragma unroll
//			for (int i = 0; i < 3; ++i)
//				out.ptr(blockIdx.x)[i] = sum[i];
//	}
//
//	int width_, N;
//	PtrStep<float> out;
//	float mean_icp_, mean_rgb_;
//	PtrStep<CorrespItem> corresp_image_;
//};
//
//__global__ void compute_weight_matrix_kernel(ComputeWeightMatrix cwm)
//{
//	cwm();
//}
//
//class ComputeJacobian
//{
//public:
//
//	__device__ __inline__ void compute_jacobian(int& k, float* value)
//	{
//		int y = k / width_;
//		int x = k - y * width_;
//
//		float row_icp[7] = { 0, 0, 0, 0, 0, 0, 0 };
//		float row_rgb[7] = { 0, 0, 0, 0, 0, 0, 0 };
//		CorrespItem& item = corresp_image_.ptr(y)[x];
//
//		float tmp0[7] = { 0, 0, 0, 0, 0, 0, 0 };
//		float tmp1[7] = { 0, 0, 0, 0, 0, 0, 0 };
//
//		if(item.valid)
//		{
//			float3 vlast = make_float3(vmap_last_.ptr(item.v)[item.u]);
//			float3 nlast = make_float3(nmap_last_.ptr(item.v)[item.u]);
//			row_icp[6] = -item.icp_residual;
//			*(float3*) &row_icp[0] = -nlast;
//			*(float3*) &row_icp[3] = cross(nlast, vlast);
//
//			float gx = (float)sobel_x_.ptr(y)[x] / 9.0;
//			float gy = (float)sobel_y_.ptr(y)[x] / 9.0;
//
//			float3 left;
//			float3 point = rcurrInv_ * (vlast - tcurr_);
//			float invz = 1.0f / point.z;
//			left.x = gx * fx_ * invz;
//			left.y = gy * fy_ * invz;
//			left.z = -(left.x * point.x + left.y * point.y) * invz;
//
//			*(float3*) &row_rgb[0] = left;
//			*(float3*) &row_rgb[3] = cross(point, left);
//			row_rgb[6] = -item.rgb_residual;
//
//#pragma unroll
//			for(int i = 0; i < 7; ++i)
//			{
//				tmp0[i] = row_icp[i] * sigma_icp_ + row_rgb[i] * cov_icp_rgb_;
//				tmp1[i] = row_icp[i] * cov_icp_rgb_ + row_rgb[i] * sigma_rgb_;
//			}
//		}
//
//		int count = 0;
//#pragma unroll
//		for (int i = 0; i < 7; ++i)
//		{
//#pragma unroll
//			for (int j = i; j < 7; ++j)
//			{
//				value[count++] = tmp0[i] * row_icp[j] + tmp1[i] * row_rgb[j];
//			}
//		}
//
//		value[count] = (float) item.valid;
//	}
//
//	__device__ __inline__ void operator()()
//	{
//		float sum[29] = { 0, 0, 0, 0, 0,
//				   	   	  0, 0, 0, 0, 0,
//				   	   	  0, 0, 0, 0, 0,
//				   	   	  0, 0, 0, 0, 0,
//				   	   	  0, 0, 0, 0, 0,
//				   	   	  0, 0, 0, 0 };
//
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		float value[29];
//		for (; i < N; i += blockDim.x * gridDim.x)
//		{
//			compute_jacobian(i, value);
//#pragma unroll
//			for (int j = 0; j < 29; ++j)
//				sum[j] += value[j];
//		}
//
//		BlockReduce<float, 29>(sum);
//
//		if (threadIdx.x == 0)
//#pragma unroll
//			for (int i = 0; i < 29; ++i)
//				out.ptr(blockIdx.x)[i] = sum[i];
//	}
//
//	int N, width_;
//	Matrix3f rcurr_, rcurrInv_;
//	float3 tcurr_;
//	float fx_, fy_, cx_, cy_;
//	PtrStep<float> out;
//	float sigma_icp_, sigma_rgb_, cov_icp_rgb_;
//	float mean_icp_, mean_rgb_;
//	PtrStep<short> sobel_x_, sobel_y_;
//	PtrStep<CorrespItem> corresp_image_;
//	PtrStep<float4> vmap_last_, nmap_last_;
//};
//
//__global__ void compute_jacobian_kernel(ComputeJacobian cj)
//{
//	cj();
//}
//
//void compute_residual_sum(DeviceArray2D<float4>& vmap_curr,
//		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& nmap_curr,
//		DeviceArray2D<float4>& nmap_last,
//		DeviceArray2D<unsigned char>& image_curr,
//		DeviceArray2D<unsigned char>& image_last, Matrix3f rcurr, float3 tcurr,
//		float* K, DeviceArray2D<float>& sum, DeviceArray<float>& out,
//		DeviceArray2D<float> & sumSE3, DeviceArray<float> & outSE3,
//		DeviceArray2D<short>& dIdx, DeviceArray2D<short>& dIdy, Matrix3f rcurrInv,
//		float * residual, double * matrixA_host, double * vectorB_host,
//		DeviceArray2D<CorrespItem>& corresp_image)
//{
//	ResidualSum rs;
//	rs.vmap_curr_ = vmap_curr;
//	rs.vmap_last_ = vmap_last;
//	rs.nmap_curr_ = nmap_curr;
//	rs.nmap_last_ = nmap_last;
//	rs.image_curr_ = image_curr;
//	rs.image_last_ = image_last;
//	rs.width_ = vmap_curr.cols;
//	rs.height_ = vmap_curr.rows;
//	rs.fx_ = K[0];
//	rs.fy_ = K[1];
//	rs.cx_ = K[2];
//	rs.cy_ = K[3];
//	rs.angle_thresh_ = 1.35f;
//	rs.dist_thresh_ = 0.2f;
//	rs.var_thresh_ = 125;
//	rs.rcurr_ = rcurr;
//	rs.tcurr_ = tcurr;
//	rs.out = sum;
//	rs.corresp_image_ = corresp_image;
//
//	compute_residual_sum_kernel<<<96, 224>>>(rs);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	Reduce<float, 3> <<<1, MaxThread>>>(sum, out, 96);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	float host_data[3];
//	out.download((void*) host_data);
//	int num_corresp_ = (int)host_data[2];
//
//	float mean_icp = host_data[0] / host_data[2];
//	float mean_rgb = host_data[1] / host_data[2];
//
//	ComputeWeightMatrix cwm;
//	cwm.corresp_image_ = corresp_image;
//	cwm.mean_icp_ = mean_icp;
//	cwm.mean_rgb_ = mean_rgb;
//	cwm.width_ = vmap_curr.cols;
//	cwm.N = vmap_curr.cols * vmap_curr.rows;
//	cwm.out = sum;
//
//	compute_weight_matrix_kernel<<<96, 224>>>(cwm);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	Reduce<float, 3> <<<1, MaxThread>>>(sum, out, 96);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	out.download((void*) host_data);
//
//	host_data[0] /= num_corresp_;
//	host_data[1] /= num_corresp_;
//	host_data[2] /= num_corresp_;
//
//	Eigen::Matrix2f cov_mat, cov_mat_inv;
//	cov_mat << host_data[0], host_data[2], host_data[2], host_data[1];
//	cov_mat_inv = cov_mat.inverse().eval();
//
//	ComputeJacobian cj;
//	cj.N = vmap_curr.cols * vmap_curr.rows;
//	cj.width_ = vmap_curr.cols;
//	cj.corresp_image_ = corresp_image;
//	cj.cov_icp_rgb_ = cov_mat_inv(0, 1);
//	cj.sigma_icp_ = cov_mat_inv(0, 0);
//	cj.sigma_rgb_ = cov_mat_inv(1, 1);
//	cj.fx_ = K[0];
//	cj.fy_ = K[1];
//	cj.cx_ = K[2];
//	cj.cy_ = K[3];
//	cj.nmap_last_ = nmap_last;
//	cj.vmap_last_ = vmap_last;
//	cj.out = sumSE3;
//	cj.rcurrInv_ = rcurrInv;
//	cj.tcurr_ = tcurr;
//	cj.sobel_x_ = dIdx;
//	cj.sobel_y_ = dIdy;
//	cj.mean_rgb_ = 0;
//
//	compute_jacobian_kernel<<<96, 224>>>(cj);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	Reduce<float, 29> <<<1, MaxThread>>>(sumSE3, outSE3, 96);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	float host_data2[29];
//	outSE3.download((void*) host_data2);
//	CreateMatrix<6, 7>(host_data2, matrixA_host, vectorB_host);
//
//	residual[0] = host_data2[27];
//	residual[1] = host_data2[28];
//}

__global__ void initialize_weight_kernel(PtrSz<float> weight)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x >= weight.size)
		return;

	weight[x] = 1.0f;
}

void initialize_weight(DeviceArray<float>& weight)
{
	dim3 block(MaxThread);
	dim3 grid(divUp(weight.size, block.x));

	initialize_weight_kernel<<<grid, block>>>(weight);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct ComputeResidualStruct
{
	const float dist_thresh = 0.1;
	const float angle_thresh = 0.6;

	__device__ __forceinline__ bool find_corresp(int& x, int& y, int& u, int& v)
	{
		float3 v_c = make_float3(vmap_curr.ptr(y)[x]);
		v_g = r * v_c + t;
		u = __float2int_rd(fx * v_g.x / v_g.z + cx + 0.5);
		v = __float2int_rd(fy * v_g.y / v_g.z + cy + 0.5);

		if (u >= 0 && v >= 0 && u < cols - 1 && v < rows - 1)
		{
			v_l = make_float3(vmap_last.ptr(v)[u]);
			n_l = make_float3(nmap_last.ptr(v)[u]);
			i_c = image_curr.ptr(y)[x];
			i_l = image_last.ptr(v)[u];
			float3 n_g = r * make_float3(nmap_curr.ptr(y)[x]);
			float dist = norm(v_g - v_l);
			float angle = norm(cross(n_l, n_g));

			return v_l == v_l && n_l == n_l && dist < dist_thresh && angle < angle_thresh;
//			return v_l == v_l && n_l == n_l && i_c > 0 && i_l > 0;
		}

		return false;
	}

	__device__ __forceinline__ void compute_residual_scale(int& k)
	{
		int y = k / cols;
		int x = k - y * cols;
		int u;
		int v;

		bool corresp_found = find_corresp(x, y, u, v);

		if(corresp_found)
		{
			Corresp& item = corresp[k];
			item.u = u;
			item.v = v;
			item.valid = true;

			ResidualVector& res = residual[k];
			res.valid = true;
			res.icp = n_l * (v_g - v_l);
			res.rgb = i_l - i_c;
		}
		else
		{
			Corresp& item = corresp[k];
			item.valid = false;

			ResidualVector& res = residual[k];
			res.valid = false;
		}
	}

	__device__ __forceinline__ void operator()()
	{
		int k = blockDim.x * blockIdx.x + threadIdx.x;
		for(; k < cols * rows; k += gridDim.x * blockDim.x)
			compute_residual_scale(k);
	}

	int rows, cols;
	float fx, fy, cx, cy;
	Matrix3f r; float3 t;
	PtrSz<ResidualVector> residual;
	PtrSz<Corresp> corresp;
	PtrSz<float> weight_map;
	PtrStep<float> image_curr, image_last;
	PtrStep<float4> vmap_curr, vmap_last;
	PtrStep<float4> nmap_curr, nmap_last;

private:
	float3 v_g, v_l, n_l;
	float i_c, i_l;
};

__global__ void compute_residual_kernel(ComputeResidualStruct crs)
{
	crs();
}

void compute_residual(DeviceArray2D<float4>& vmap_curr,
		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& nmap_curr,
		DeviceArray2D<float4>& nmap_last,
		DeviceArray2D<float>& image_curr,
		DeviceArray2D<float>& image_last, DeviceArray<float>& weight,
		DeviceArray<ResidualVector>& residual, DeviceArray<Corresp>& corresp,
		Matrix3f r, float3 t, float* intrinsics)
{
	int cols = vmap_curr.cols;
	int rows = vmap_curr.rows;

	ComputeResidualStruct crs;
	crs.vmap_curr = vmap_curr;
	crs.vmap_last = vmap_last;
	crs.nmap_curr = nmap_curr;
	crs.nmap_last = nmap_last;
	crs.image_curr = image_curr;
	crs.image_last = image_last;
	crs.weight_map = weight;
	crs.corresp = corresp;
	crs.r = r;
	crs.t = t;
	crs.fx = intrinsics[0];
	crs.fy = intrinsics[1];
	crs.cx = intrinsics[2];
	crs.cy = intrinsics[3];
	crs.residual = residual;
	crs.cols = cols;
	crs.rows = rows;

	compute_residual_kernel<<<96, 224>>>(crs);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct ComputeScaleStruct
{
	__device__ __forceinline__ void compute_scale(int& k, float* sum)
	{
		ResidualVector& r = residual[k];
		float& w = weight[k];
		if(r.valid)
		{
			sum[0] = w * r.icp * r.icp;
			sum[1] = w * r.icp * r.rgb;
			sum[2] = w * r.rgb * r.rgb;
			sum[3] = 1;
		}
	}

	__device__ __forceinline__ void operator()()
	{
		float sum[4] = { 0, 0, 0, 0 };
		float val[4] = { 0, 0, 0, 0 };
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		for(; i < N; i += gridDim.x * blockDim.x)
		{
			compute_scale(i, val);
			for(int j = 0; j < 4; ++j)
				sum[j] += val[j];
		}

		BlockReduce<float, 4>(sum);

		if (threadIdx.x == 0)
#pragma unroll
			for (int i = 0; i < 4; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}

	int N;
	PtrSz<ResidualVector> residual;
	PtrSz<float> weight;
	PtrStep<float> out;
};

__global__ void compute_scale_kernel(ComputeScaleStruct css)
{
	css();
}

Eigen::Matrix<float, 2, 2> compute_scale(DeviceArray<ResidualVector>& residual,
		DeviceArray<float>& weight, DeviceArray2D<float>& sum,
		DeviceArray<float>& out, int N, float& point_ratio)
{
	ComputeScaleStruct css;
	css.residual = residual;
	css.weight = weight;
	css.out = sum;
	css.N = N;

	compute_scale_kernel<<<96, 224>>>(css);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	Reduce<float, 4><<<1, MaxThread>>>(sum, out, 96);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[4];
	out.download((void*) host_data);
	Eigen::Matrix<float, 2, 2> sigma;
	sigma << host_data[0], host_data[1], host_data[1], host_data[2];
	sigma /= (host_data[3] + 1);
	point_ratio = host_data[3] / N;
	return sigma.reverse();
}

struct ComputeWeightStruct
{
	__device__ __forceinline__ void compute_weight(int& k)
	{
		float& w = weight[k];
		ResidualVector& r = residual[k];
		float tmp = r.icp * r.icp * scale.x;
		tmp += 2 * r.icp * r.rgb * scale.y;
		tmp += r.rgb * r.rgb * scale.z;
		w = (2 + 5) / (5 + tmp);
	}

	__device__ __forceinline__ void operator()()
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		compute_weight(i);
	}

	int cols, rows;
	PtrSz<ResidualVector> residual;
	PtrSz<float> weight;
	float3 scale;
};

__global__ void compute_weight_kernel(ComputeWeightStruct cws)
{
	cws();
}

void compute_weight(DeviceArray<ResidualVector>& residual,
		DeviceArray<float>& weight, float3 scale)
{
	ComputeWeightStruct cws;
	cws.residual = residual;
	cws.weight = weight;
	cws.scale = scale;

	dim3 thread(MaxThread);
	dim3 block(divUp(residual.size, thread.x));

	compute_weight_kernel<<<thread, block>>>(cws);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct ComputeLeastSquareStruct
{
	__device__ __forceinline__ void compute_jacobian(int& k, float* sum)
	{
		float icp_row[7] = { 0, 0, 0, 0, 0, 0, 0 };
		float rgb_row[7] = { 0, 0, 0, 0, 0, 0, 0 };
		float tmp_row0[7] = { 0, 0, 0, 0, 0, 0, 0 };
		float tmp_row1[7] = { 0, 0, 0, 0, 0, 0, 0 };

		Corresp& item = corresp[k];

		if(item.valid)
		{
			int y = k / cols;
			int x = k - y * cols;

			ResidualVector& res = residual[k];
			icp_row[6] = -res.icp;
			rgb_row[6] = -res.rgb;

//			float3 vlast = r * make_float3(vmap_curr.ptr(y)[x]) + t;
			float3 vlast = make_float3(vmap.ptr(item.v)[item.u]);
			float3 nlast = make_float3(nmap.ptr(item.v)[item.u]);

			if(vlast == vlast && nlast == nlast)
			{
				*(float3*)&icp_row[0] = nlast;
				*(float3*)&icp_row[3] = cross(vlast, nlast);

				float gx = dIdx.ptr(item.v)[item.u];
				float gy = dIdy.ptr(item.v)[item.u];

				float3 left;
				float3 point = vlast;
//				float3 point = make_float3(vmap_curr.ptr(y)[x]);
				float z_inv = 1.0f / point.z;

				left.x = gx * fx * z_inv;
				left.y = gy * fy * z_inv;
				left.z = -(left.x * point.x + left.y * point.y) * z_inv;

				*(float3*) &rgb_row[0] = left;
				*(float3*) &rgb_row[3] = cross(point, left);

	#pragma unroll
				for(int i = 0; i < 7; ++i)
				{
					tmp_row0[i] = icp_row[i] * scale.x + rgb_row[i] * scale.y;
					tmp_row1[i] = icp_row[i] * scale.y + rgb_row[i] * scale.z;
				}
			}
		}

		int count = 0;
#pragma unroll
		for (int i = 0; i < 7; ++i)
		{
#pragma unroll
			for (int j = i; j < 7; ++j)
			{
				sum[count++] = tmp_row0[i] * icp_row[j] + tmp_row1[i] * rgb_row[j];
//				sum[count++] = rgb_row[i] * rgb_row[j];
//				sum[count++] = icp_row[i] * icp_row[j];
			}
		}

		sum[count] = (float) item.valid;
	}

	__device__ __forceinline__ void operator()()
	{
		float sum[29] = { 0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0 };

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		float val[29];
		for (; i < cols * rows; i += blockDim.x * gridDim.x)
		{
			compute_jacobian(i, val);
#pragma unroll
			for (int j = 0; j < 29; ++j)
				sum[j] += val[j];
		}

		BlockReduce<float, 29>(sum);

		if (threadIdx.x == 0)
#pragma unroll
			for (int i = 0; i < 29; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];

	}

	float fx, fy;
	float3 scale;
	PtrSz<Corresp> corresp;
	PtrSz<ResidualVector> residual;
	PtrStep<float> out;
	PtrStep<float4> vmap, vmap_curr, nmap;
	int cols, rows;
	PtrStep<float> dIdx, dIdy;
	Matrix3f r, r_inv; float3 t;
};

__global__ void compute_least_square_kernel(ComputeLeastSquareStruct clss)
{
	clss();
}

void compute_least_square(DeviceArray<Corresp>& corresp,
		DeviceArray<ResidualVector>& residual_vec, DeviceArray<float>& weight,
		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& vmap_curr,
		DeviceArray2D<float4>& nmap_last, DeviceArray2D<float>& dIdx,
		DeviceArray2D<float>& dIdy, DeviceArray2D<float>& sum,
		DeviceArray<float>& out, Matrix3f r, Matrix3f r_inv, float3 t,
		float3 scale, float* intrinsics, double* matrixA_host,
		double* vectorB_host, float* residual)
{
	ComputeLeastSquareStruct clss;
	clss.vmap = vmap_last;
	clss.vmap_curr = vmap_curr;
	clss.nmap = nmap_last;
	clss.dIdx = dIdx;
	clss.dIdy = dIdy;
	clss.corresp = corresp;
	clss.residual = residual_vec;
	clss.fx = intrinsics[0];
	clss.fy = intrinsics[1];
	clss.out = sum;
	clss.cols = vmap_last.cols;
	clss.rows = vmap_last.rows;
	clss.scale = scale;
	clss.r_inv = r_inv;
	clss.r = r;
	clss.t = t;

	compute_least_square_kernel<<<96, 224>>>(clss);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	Reduce<float, 29> <<<1, MaxThread>>>(sum, out, 96);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	out.download((void*) host_data);
	CreateMatrix<6, 7>(host_data, matrixA_host, vectorB_host);

	residual[0] = host_data[27];
	residual[1] = host_data[28];
}

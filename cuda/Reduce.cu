#include "Frame.hpp"
#include "DeviceMath.h"
#include "DeviceFunc.h"
#include "DeviceArray.h"

template<typename T, int size> __device__
inline void WarpReduceSum(T* val) {
	for(int offset = WarpSize / 2; offset > 0; offset /= 2) {
		for(int i = 0; i < size; ++i) {
			val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
		}
	}
}

template<typename T, int size> __device__
inline void BlockReduceSum(T* val) {
	static __shared__ T shared[32 * size];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	WarpReduceSum<T, size>(val);

    if(lane == 0)
    	memcpy(&shared[wid * size], val, sizeof(T) * size);

    __syncthreads();

    if(threadIdx.x < blockDim.x / warpSize)
    	memcpy(val, &shared[lane * size], sizeof(T) * size);
    else
    	memset(val, 0, sizeof(T) * size);

    if(wid == 0)
        WarpReduceSum<T, size>(val);
}

template<typename T, int size> __global__
void ReduceSum(PtrStep<T> in, T* out, int N) {
	T sum[size];
	memset(sum, 0, sizeof(T) * size);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(; i < N; i += blockDim.x * gridDim.x)
		for(int j = 0; j < size; ++j)
			sum[j] += in.ptr(i)[j];

    BlockReduceSum<T, size>(sum);

    if(threadIdx.x == 0)
		for(int i = 0; i < size; ++i)
			out[i] = sum[i];
}

struct ICPReduce {

	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	float3 tcurr;
	float3 tlast;
	PtrStep<float4> VMapCurr, VMapLast;
	PtrStep<float3> NMapCurr, NMapLast;
	int cols, rows, N;
	float fx, fy, cx, cy;
	float angleThresh, distThresh;

	mutable PtrStepSz<float> out;

	__device__ inline
	bool SearchPoint(int& x, int& y, float3& vcurr_g,
								 float3& vlast_g, float3& nlast_g) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		int u = (int)(vcurr_p.x * invz * fx + cx + 0.5);
		int v = (int)(vcurr_p.y * invz * fy + cy + 0.5);
		if(u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		float3 vlast_c = make_float3(VMapLast.ptr(v)[u]);
		vlast_g = Rlast * vlast_c + tlast;

		float3 ncurr_c = NMapCurr.ptr(y)[x];
		float3 ncurr_g = Rcurr * ncurr_c;

		float3 nlast_c = NMapLast.ptr(v)[u];
		nlast_g = Rlast * nlast_c;

		float dist = norm(vlast_g - vcurr_g);
		float sine = norm(cross(ncurr_g, nlast_g));

		return (sine < angleThresh && dist <= distThresh &&
					!isnan(ncurr_c.x) && !isnan(nlast_c.x));
	}

	__device__ inline
	void GetRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);

		bool found = false;
		float3 vcurr, vlast, nlast;
		found = SearchPoint(x, y, vcurr, vlast, nlast);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if(found) {
			nlast = invRlast * nlast;
			vcurr = invRlast * (vcurr - tlast);
			vlast = invRlast * (vlast - tlast);
			*(float3*)&row[0] = -nlast;
			*(float3*)&row[3] = cross(nlast, vlast);
            row[6] = -nlast * (vlast - vcurr);
		}

		int count = 0;
		for (int i = 0; i < 7; ++i)
			for (int j = i; j < 7; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float)found;
	}

	template<typename T, int size>
	__device__ void operator()() const {
		T sum[size];
		T val[size];
		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			GetRow(i, val);

			for(int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		BlockReduceSum<T, size>(sum);

		if (threadIdx.x == 0)
			for(int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void
ICPReduceSum_device(const ICPReduce icp) {
	icp.template operator()<float, 29>();
}

static void inline
CreateMatrix(float* host_data, float* host_a, float* host_b) {
    int shift = 0;
	for (int i = 0; i < 6; ++i)
		for (int j = i; j < 7; ++j) {
			float value = host_data[shift++];
			if (j == 6)
				host_b[i] = value;
			else
				host_a[j * 6 + i] = host_a[i * 6 + j] = value;
		}
}

float ICPReduceSum(Frame& NextFrame, Frame& LastFrame,
								   int pyr, float* host_a, float* host_b) {

	DeviceArray2D<float> sum(29, 96);
	DeviceArray<float> result(29);
	result.zero();
	sum.zero();

	ICPReduce icp;
	icp.out = sum;
	icp.VMapCurr = NextFrame.mVMap[pyr];
	icp.NMapCurr = NextFrame.mNMap[pyr];
	icp.VMapLast = LastFrame.mVMap[pyr];
	icp.NMapLast = LastFrame.mNMap[pyr];
	icp.cols = Frame::cols(pyr);
	icp.rows = Frame::rows(pyr);
	icp.N = Frame::pixels(pyr);
	icp.Rcurr = NextFrame.Rot_gpu();
	icp.tcurr = NextFrame.Trans_gpu();
	icp.Rlast = LastFrame.Rot_gpu();
	icp.invRlast = LastFrame.RotInv_gpu();
	icp.tlast = LastFrame.Trans_gpu();
	icp.angleThresh = 0.6;
	icp.distThresh = 0.1;
	icp.fx = Frame::fx(pyr);
	icp.fy = Frame::fy(pyr);
	icp.cx = Frame::cx(pyr);
	icp.cy = Frame::cy(pyr);

	ICPReduceSum_device<<<96, 224>>>(icp);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	ReduceSum<float, 29><<<1, MaxThread>>>(sum, result, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	result.download(host_data);
	CreateMatrix(host_data, host_a, host_b);
	return sqrt(host_data[27]) / host_data[28];
}

struct RGBReduce {

	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	float3 tcurr;
	float3 tlast;
	PtrStep<float> dIx, dIy;
	PtrStep<uchar> GrayCurr, GrayLast;
	PtrStep<float4> VMapCurr, VMapLast;
	int cols, rows, N;
	float fx, fy, cx, cy, minGxy;

	mutable PtrStep<uchar> corres;
	mutable PtrStepSz<float> out;

	__device__ inline
	bool SearchPoint(int& x, int& y, float& gx, float& gy, float& diff, float3& vlast_c) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		float3 vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		int u = (int)(vcurr_p.x * invz * fx + cx + 0.5);
		int v = (int)(vcurr_p.y * invz * fy + cy + 0.5);
		if(u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		vlast_c = make_float3(VMapLast.ptr(v)[u]);
		if(isnan(vlast_c.x) || vlast_c.z < 1e-3)
			return false;

		float3 vlast_g = Rlast * vlast_c + tlast;

		gx = dIx.ptr(v)[u];
		gy = dIy.ptr(v)[u];
		diff = (float)GrayLast.ptr(v)[u] - (float)GrayCurr.ptr(y)[x];

		return (vlast_g.z - vcurr_g.z) < 0.6 && (gx * gx + gy * gy) >= minGxy;
	}

	__device__ inline
	void GetRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);

		float3 vlast;
		float gx, gy, diff;
		bool found = false;
		found = SearchPoint(x, y, gx, gy, diff, vlast);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if(found) {
			float3 dIdh;
			float invz = 1.0 / vlast.z;
			dIdh.x = gx * fx * invz;
			dIdh.y = gy * fy * invz;
			dIdh.z = -(dIdh.x * vlast.x + dIdh.y * vlast.y) * invz;
			*(float3*)&row[0] = dIdh;
			*(float3*)&row[3] = -cross(dIdh, vlast);
            row[6] = -diff;
            corres.ptr(y)[x] = 255;
		}

		int count = 0;
		for (int i = 0; i < 7; ++i)
			for (int j = i; j < 7; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float)found;
	}

	template<typename T, int size>
	__device__ void operator()() const {
		T sum[size];
		T val[size];
		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			GetRow(i, val);

			for(int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		BlockReduceSum<T, size>(sum);

		if (threadIdx.x == 0)
			for(int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void
RGBReduceSum_device(const RGBReduce rgb) {
	rgb.template operator()<float, 29>();
}

float RGBReduceSum(Frame& NextFrame, Frame& LastFrame,
								   int pyr, float* host_a, float* host_b) {

	DeviceArray2D<uchar> Corres(Frame::cols(pyr), Frame::rows(pyr));
	DeviceArray2D<float> sum(29, 96);
	DeviceArray<float> result(29);
	Corres.zero();
	result.zero();
	sum.zero();

	float minGxy[3] = { 25, 9, 1 };

	RGBReduce rgb;
	rgb.out = sum;
	rgb.corres = Corres;
	rgb.minGxy = minGxy[pyr];
	rgb.dIx = LastFrame.mdIx[pyr];
	rgb.dIy = LastFrame.mdIy[pyr];
	rgb.GrayCurr = NextFrame.mGray[pyr];
	rgb.VMapCurr = NextFrame.mVMap[pyr];
	rgb.GrayLast = LastFrame.mGray[pyr];
	rgb.VMapLast = LastFrame.mVMap[pyr];
	rgb.cols = Frame::cols(pyr);
	rgb.rows = Frame::rows(pyr);
	rgb.N = Frame::pixels(pyr);
	rgb.Rcurr = NextFrame.Rot_gpu();
	rgb.tcurr = NextFrame.Trans_gpu();
	rgb.Rlast = LastFrame.Rot_gpu();
	rgb.invRlast = LastFrame.RotInv_gpu();
	rgb.tlast = LastFrame.Trans_gpu();
	rgb.fx = Frame::fx(pyr);
	rgb.fy = Frame::fy(pyr);
	rgb.cx = Frame::cx(pyr);
	rgb.cy = Frame::cy(pyr);

	RGBReduceSum_device<<<96, 224>>>(rgb);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	ReduceSum<float, 29><<<1, MaxThread>>>(sum, result, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat hostcorres(Frame::rows(pyr), Frame::cols(pyr), CV_8UC1);
	Corres.download((void*)hostcorres.data, hostcorres.step);
	cv::imshow("corresp", hostcorres);
	cv::imwrite("corresp.jpg", hostcorres);

	float host_data[29];
	result.download(host_data);
	CreateMatrix(host_data, host_a, host_b);
	return sqrt(host_data[27]) / host_data[28];
}

struct RGB {

	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	float3 tcurr;
	float3 tlast;
	PtrStep<float> dIx, dIy;
	PtrStep<uchar> GrayCurr, GrayLast;
	PtrStep<float4> VMapCurr, VMapLast;
	int cols, rows, N;
	float fx, fy, cx, cy, minGxy;

	mutable PtrStep<float> residual;
	mutable PtrStep<int2> coord;
	mutable PtrStep<bool> found;
	mutable PtrStep<float> out;

	__device__ bool
	FindCorresp(int& x, int& y, int& u, int& v, float& gx,
						 float& gy, float& diff, float3& vlast_c) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		float3 vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		u = (int)(vcurr_p.x * invz * fx + cx + 0.5);
		v = (int)(vcurr_p.y * invz * fy + cy + 0.5);
		if(u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		vlast_c = make_float3(VMapLast.ptr(v)[u]);
		if(isnan(vlast_c.x) || vlast_c.z < 1e-3)
			return false;

		float3 vlast_g = Rlast * vlast_c + tlast;

		gx = dIx.ptr(v)[u];
		gy = dIy.ptr(v)[u];
		diff = (float)GrayLast.ptr(v)[u] - (float)GrayCurr.ptr(y)[x];

		return (vlast_g.z - vcurr_g.z) < 0.6 && (gx * gx + gy * gy) >= minGxy;
	}

	__device__ inline
	void GetRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);

		int u, v;
		float3 vlast;
		float gx, gy, diff;
		bool found_correp = false;
		found_correp= FindCorresp(x, y, u, v, gx, gy, diff, vlast);

		if(found_correp) {

			residual.ptr(y)[x] = diff;
			found.ptr(y)[x] = found_correp;
			coord.ptr(y)[x] = { u, v };
			sum[0] = diff;
			sum[1] = 1;
		}
	}

	template<typename T, int size>
	__device__ void operator()() const {
		T sum[size];
		T val[size];
		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			GetRow(i, val);

			for(int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		BlockReduceSum<T, size>(sum);

		if (threadIdx.x == 0)
			for(int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void
RGBDense_device(const RGB rgb) {
	rgb.template operator()<float, 2>();
}

void RGBDense(Frame& NextFrame, Frame& LastFrame) {

}

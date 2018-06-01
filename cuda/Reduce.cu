#include "Frame.h"
#include "Converter.h"
#include "DeviceMath.h"
#include "DeviceFunc.h"
#include "DeviceArray.h"

template<typename T, int size> __device__
inline void WarpReduceSum(T* val) {
#pragma unroll
	for(int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
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
#pragma unroll
		for(int j = 0; j < size; ++j)
			sum[j] += in.ptr(i)[j];

    BlockReduceSum<T, size>(sum);

    if(threadIdx.x == 0)
#pragma unroll
		for(int i = 0; i < size; ++i)
			out[i] = sum[i];
}

struct ICPReduce {

	bool ICPOnly;
	float icpW;
	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	Matrix3f R;
	float3 t;
	float3 tcurr;
	float3 tlast;
	PtrStep<float> dIx, dIy;
	PtrStep<float4> VMapCurr, VMapLast;
	PtrStep<float3> NMapCurr, NMapLast;
	PtrStep<uchar> GrayCurr, GrayLast;
	mutable PtrStep<uchar> Corresp;
	int cols, rows, N;
	float fx, fy, cx, cy, minGxy;
	float angleThresh, distThresh;

	mutable float w;
	mutable PtrStepSz<float> out;

	__device__ inline
	bool SearchCorresp(bool& view, int& x, int& y, int& u, int& v, float3& vcurr_g,
					   float3& vlast_g, float3& nlast_g, float3& vcurr_p) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = Rcurr * vcurr_c + tcurr;
		vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		u = __float2int_rd(vcurr_p.x * invz * fx + cx + 0.5);
		v = __float2int_rd(vcurr_p.y * invz * fy + cy + 0.5);
		if(u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		view = true;
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
	bool ComputeRGB(int& x, int& y, int& u, int& v,
								   float3& vcg, float3& vcp, float3& vlast, float* row) const {

		float dx = dIx.ptr(v)[u];
		float dy = dIy.ptr(v)[u];
		if(dx * dx + dy * dy < minGxy)
			return false;

		if(abs(vlast.z - vcg.z) > 0.5)
			return false;

		bool valid = GrayLast.ptr(v)[u] > 0;
		const int r = 2;
		for(int i = max(0, x - r ); i < min(x + r + 1, cols); ++i)
			for(int j = max(0, y - r ); j < min(y + r + 1, rows); ++j)
				valid = (GrayCurr.ptr(j)[i] > 0) && valid;

		if(!valid)
			return false;

		float w = abs((float)GrayCurr.ptr(y)[x] - (float)GrayLast.ptr(v)[u]);
		w = w > 1e-7? 1.0 / w : 1.0;
		dx *= w;
		dy *= w;

		float3 rcx = invRlast.coloumx();
		float3 rcy = invRlast.coloumy();
		float3 rcz = invRlast.coloumz();
		float3 dIdh;
		dIdh.x =  -dx * fx / vcp.z;
		dIdh.y =  -dy * fy / vcp.z;
		dIdh.z = (dx * fx * vcp.x / vcp.z + dy * fy * vcp.y / vcp.z) /  vcp.z;
		float3 r0xp = -cross(invRlast.rowx, vcg);
		float3 r1xp = -cross(invRlast.rowy, vcg);
		float3 r2xp = -cross(invRlast.rowz, vcg);

		row[0] = dIdh * rcx;
		row[1] = dIdh * rcy;
		row[2] = dIdh * rcz;
		row[3] = dIdh * make_float3(r0xp.x, r1xp.x, r2xp.x);
		row[4] = dIdh * make_float3(r0xp.y, r1xp.y, r2xp.y);
		row[5] = dIdh * make_float3(r0xp.z, r1xp.z, r2xp.z);
		row[6] = -w * ((float)GrayCurr.ptr(y)[x] - (float)GrayLast.ptr(v)[u]);

		return true;
	}

	template<bool bICPOnly> __device__ inline
	void GetRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);
		Corresp.ptr(y)[x] = 0;

		int u = 0, v = 0;
		bool bCorresp = false;
		bool bRGB = false;
		bool bView = false;
		float3 vcurr, vlast, nlast, vcurrp;
		bCorresp = SearchCorresp(bView, x, y, u, v, vcurr, vlast, nlast, vcurrp);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };
		float row_rgb[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if(bCorresp) {
			float3 nvcross = cross(nlast, vcurr);
            row[0] = -nlast.x;
            row[1] = -nlast.y;
            row[2] = -nlast.z;
            row[3] = nvcross.x;
            row[4] = nvcross.y;
            row[5] = nvcross.z;
            row[6] = -nlast * (vlast - vcurr);
		}

        if(bView && !bICPOnly) {
        	bRGB = ComputeRGB(x, y, u, v, vcurr, vcurrp, vlast, row_rgb);
        	if(bRGB)
        		Corresp.ptr(y)[x] = 255;
        }

		int count = 0;
		if(bICPOnly || !bRGB) {
#pragma unroll
			for(int i = 0; i < 7; ++i)
#pragma unroll
				for(int j = i; j < 7; ++j)
//					sum[count++] = row[i] * row[j];
					sum[count++] = row_rgb[i] * row_rgb[j];
		}
		else {
#pragma unroll
			for(int i = 0; i < 7; ++i)
#pragma unroll
				for(int j = i; j < 7; ++j)
//					sum[count++] = icpW * row[i] * row[j] + (1 - icpW) * row_rgb[i] * row_rgb[j];
					sum[count++] = row_rgb[i] * row_rgb[j];
//					sum[count++] = row[i] * row[j];
		}
		sum[count] = (float)(bCorresp || bRGB);
	}

	template<typename T, int size, bool bRGB>
	__device__ void operator()() const {
		T sum[size];
		T val[size];
		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			GetRow<bRGB>(i, val);

#pragma unroll
			for(int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		BlockReduceSum<T, size>(sum);

		if (threadIdx.x == 0)
#pragma unroll
			for(int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

template<bool bICPOnly> __global__ void
ICPReduceSum_device(const ICPReduce icp) {
	icp.template operator()<float, 29, bICPOnly>();
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

void ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int pyr,
				  float* host_a, float* host_b, float& cost) {

	DeviceArray2D<uchar> Corresp(Frame::cols(pyr), Frame::rows(pyr));
	DeviceArray2D<float> sum(29, 96);
	DeviceArray<float> result(29);
	Corresp.zero();
	result.zero();
	sum.zero();

	float minGxy[Frame::numPyrs] = { 25, 9, 1 };

	ICPReduce icp;
	icp.out = sum;
	icp.minGxy = minGxy[pyr];
	icp.Corresp = Corresp;
	icp.dIx = LastFrame.mdIx[pyr];
	icp.dIy = LastFrame.mdIy[pyr];
	icp.VMapCurr = NextFrame.mVMap[pyr];
	icp.NMapCurr = NextFrame.mNMap[pyr];
	icp.GrayCurr = NextFrame.mGray[pyr];
	icp.VMapLast = LastFrame.mVMap[pyr];
	icp.NMapLast = LastFrame.mNMap[pyr];
	icp.GrayLast = LastFrame.mGray[pyr];
	icp.cols = Frame::cols(pyr);
	icp.rows = Frame::rows(pyr);
	icp.N = Frame::pixels(pyr);
	icp.Rcurr = NextFrame.mRcw;
	icp.tcurr = Converter::CvMatToFloat3(NextFrame.mtcw);
	icp.Rlast = LastFrame.mRcw;
	icp.invRlast = LastFrame.mRwc;
	icp.tlast = Converter::CvMatToFloat3(LastFrame.mtcw);
	icp.angleThresh = 0.6;
	icp.distThresh = 0.1;
	icp.icpW = 0;
	icp.ICPOnly = false;
	icp.fx = Frame::fx(pyr);
	icp.fy = Frame::fy(pyr);
	icp.cx = Frame::cx(pyr);
	icp.cy = Frame::cy(pyr);

	ICPReduceSum_device<false><<<96, 224>>>(icp);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	ReduceSum<float, 29><<<1, MaxThread>>>(sum, result, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::Mat corr(Frame::rows(pyr), Frame::cols(pyr), CV_8UC1);
	Corresp.download((void*)corr.data, corr.step);
	cv::imshow("Corresp", corr);

	float host_data[29];
	result.download(host_data);
	CreateMatrix(host_data, host_a, host_b);
	cost = sqrt(host_data[27]) / host_data[28];
}

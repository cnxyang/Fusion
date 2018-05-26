#include "Frame.h"
#include "Converter.h"
#include "DeviceMath.h"
#include "DeviceFunc.h"
#include "DeviceArray.h"

#define WarpSize 32
#define MaxThread 1024

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
	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	Matrix3f R;
	float3 t;
	float3 tcurr;
	float3 tlast;
	PtrStep<float4> VMapCurr, VMapLast;
	PtrStep<float3> NMapCurr, NMapLast;
	int cols, rows, N;
	float fx, fy, cx, cy;
	float angleThresh, distThresh;
	mutable PtrStepSz<float> out;

	__device__ inline
	bool SearchCorresp(int& x, int& y, float3& vcurr_g, float3& vlast_g, float3& nlast_g) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		int u = __float2int_rd(vcurr_p.x * invz * fx + cx + 0.5);
		int v = __float2int_rd(vcurr_p.y * invz * fy + cy + 0.5);
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

		return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_c.x) && !isnan(nlast_c.x));
	}

	__device__ inline
	void GetRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);

		bool bCorresp = false;
		float3 vcurr, vlast, nlast;
		bCorresp = SearchCorresp(x, y, vcurr, vlast, nlast);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if(bCorresp) {
			float3 nvcross = cross(vcurr, nlast);
            row[0] = nlast.x;
            row[1] = nlast.y;
            row[2] = nlast.z;
            row[3] = nvcross.x;
            row[4] = nvcross.y;
            row[5] = nvcross.z;
            row[6] = nlast * (vcurr - vlast);
		}

		int count = 0;
#pragma unroll
		for(int i = 0; i < 7; ++i)
#pragma unroll
			for(int j = i; j < 7; ++j) {
				sum[count++] = row[i] * row[j];
			}
		sum[count] = (float)bCorresp;
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

void ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int PyrLevel, float* host_a, float* host_b) {

	DeviceArray2D<float> sum(29, 96);
	DeviceArray<float> result(29);
	result.zero();
	sum.zero();

	ICPReduce icp;
	icp.out = sum;
	icp.VMapCurr = NextFrame.mVMap[PyrLevel];
	icp.NMapCurr = NextFrame.mNMap[PyrLevel];
	icp.VMapLast = LastFrame.mVMap[PyrLevel];
	icp.NMapLast = LastFrame.mNMap[PyrLevel];
	icp.cols = Frame::mPyrRes[PyrLevel].first;
	icp.rows = Frame::mPyrRes[PyrLevel].second;
	icp.N = Frame::N[PyrLevel];
	icp.Rcurr = NextFrame.mRcw;
	icp.tcurr = Converter::CvMatToFloat3(NextFrame.mtcw);
	icp.Rlast = LastFrame.mRcw;
	icp.invRlast = LastFrame.mRwc;
	icp.tlast = Converter::CvMatToFloat3(LastFrame.mtcw);
	icp.angleThresh = 0.6;
	icp.distThresh = 0.1;
	icp.fx = Frame::fx(PyrLevel);
	icp.fy = Frame::fy(PyrLevel);
	icp.cx = Frame::cx(PyrLevel);
	icp.cy = Frame::cy(PyrLevel);

	ICPReduceSum_device<<<96, 224>>>(icp);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	ReduceSum<float, 29><<<1, MaxThread>>>(sum, result, 96);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	result.download(host_data);
	CreateMatrix(host_data, host_a, host_b);
	std::cout << host_data[27] << std::endl;
}

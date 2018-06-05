#include "Frame.h"
#include "Converter.h"
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
	bool SearchPoint(int& x, int& y, float3& vcurr_g, float3& vlast_g, float3& nlast_g) const {

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
	icp.Rcurr = NextFrame.mRcw;
	icp.tcurr = Converter::CvMatToFloat3(NextFrame.mtcw);
	icp.Rlast = LastFrame.mRcw;
	icp.invRlast = LastFrame.mRwc;
	icp.tlast = Converter::CvMatToFloat3(LastFrame.mtcw);
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
	rgb.Rcurr = NextFrame.mRcw;
	rgb.tcurr = Converter::CvMatToFloat3(NextFrame.mtcw);
	rgb.Rlast = LastFrame.mRcw;
	rgb.invRlast = LastFrame.mRwc;
	rgb.tlast = Converter::CvMatToFloat3(LastFrame.mtcw);
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

	float host_data[29];
	result.download(host_data);
	CreateMatrix(host_data, host_a, host_b);
	return sqrt(host_data[27]) / host_data[28];
}

//#include "Frame.h"
//#include "Converter.h"
//#include "DeviceMath.h"
//#include "DeviceFunc.h"
//#include "DeviceArray.h"
//
//template<typename T, int size> __device__
//inline void WarpReduceSum(T* val) {
//#pragma unroll
//	for(int offset = WarpSize / 2; offset > 0; offset /= 2) {
//#pragma unroll
//		for(int i = 0; i < size; ++i) {
//			val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
//		}
//	}
//}
//
//template<typename T, int size> __device__
//inline void BlockReduceSum(T* val) {
//	static __shared__ T shared[32 * size];
//	int lane = threadIdx.x % warpSize;
//	int wid = threadIdx.x / warpSize;
//
//	WarpReduceSum<T, size>(val);
//
//    if(lane == 0)
//    	memcpy(&shared[wid * size], val, sizeof(T) * size);
//
//    __syncthreads();
//
//    if(threadIdx.x < blockDim.x / warpSize)
//    	memcpy(val, &shared[lane * size], sizeof(T) * size);
//    else
//    	memset(val, 0, sizeof(T) * size);
//
//    if(wid == 0)
//        WarpReduceSum<T, size>(val);
//}
//
//template<typename T, int size> __global__
//void ReduceSum(PtrStep<T> in, T* out, int N) {
//	T sum[size];
//	memset(sum, 0, sizeof(T) * size);
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//    for(; i < N; i += blockDim.x * gridDim.x)
//#pragma unroll
//		for(int j = 0; j < size; ++j)
//			sum[j] += in.ptr(i)[j];
//
//    BlockReduceSum<T, size>(sum);
//
//    if(threadIdx.x == 0)
//#pragma unroll
//		for(int i = 0; i < size; ++i)
//			out[i] = sum[i];
//}
//
//struct ICPReduce {
//
//	float icpW;
//	Matrix3f Rcurr;
//	Matrix3f Rlast;
//	Matrix3f invRlast;
//	Matrix3f R;
//	float3 t;
//	float3 tcurr;
//	float3 tlast;
//	PtrStep<float> dIx, dIy;
//	PtrStep<float4> VMapCurr, VMapLast;
//	PtrStep<float3> NMapCurr, NMapLast;
//	PtrStep<uchar> GrayCurr, GrayLast;
//	mutable PtrStep<uchar> Corresp;
//	int cols, rows, N;
//	float fx, fy, cx, cy, minGxy;
//	float angleThresh, distThresh;
//
//	mutable PtrStepSz<float> out;
//
//	__device__ inline
//	bool SearchCorresp(bool& view, int& x, int& y, int& u, int& v, float3& vcurr_g,
//					   float3& vlast_g, float3& nlast_g, float3& vcurr_p) const {
//
//		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
//		if(isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
//			return false;
//
//		vcurr_g = Rcurr * vcurr_c + tcurr;
//		vcurr_p = invRlast * (vcurr_g - tlast);
//
//		float invz = 1.0 / vcurr_p.z;
//		u = (int)(vcurr_p.x * invz * fx + cx + 0.5);
//		v = (int)(vcurr_p.y * invz * fy + cy + 0.5);
//		if(u < 0 || v < 0 || u >= cols || v >= rows)
//			return false;
//
//		view = true;
//		float3 vlast_c = make_float3(VMapLast.ptr(v)[u]);
//		vlast_g = Rlast * vlast_c + tlast;
//
//		float3 ncurr_c = NMapCurr.ptr(y)[x];
//		float3 ncurr_g = Rcurr * ncurr_c;
//
//		float3 nlast_c = NMapLast.ptr(v)[u];
//		nlast_g = Rlast * nlast_c;
//
//		float dist = norm(vlast_g - vcurr_g);
//		float sine = norm(cross(ncurr_g, nlast_g));
//
//		return (sine < angleThresh && dist <= distThresh &&
//					!isnan(ncurr_c.x) && !isnan(nlast_c.x));
//	}
//
//	template<typename T> __device__ inline
//	T interp(T val1, T val2, float x0) const {
//		return (T)((1 - x0) * val1 + x0 * val2);
//	}
//
//	template<typename T> __device__ inline
//	T interp2(float2 val, const PtrStep<T>& dI) const {
//		float x0 = val.x - (int)floor(val.x);
//		float y0 = val.y - (int)floor(val.y);
//		int u0 = (int)floor(val.x);
//		int u1 = (int)ceil(val.x);
//		int v0 = (int)floor(val.y);
//		int v1 = (int)ceil(val.y);
//		T g00 = dI.ptr(v0)[u0];
//		T g01 = dI.ptr(v0)[u1];
//		T g10 = dI.ptr(v1)[u0];
//		T g11 = dI.ptr(v1)[u1];
//		if((g00 - g01) < 1e-3)
//			return g00;
//		T gx0 = interp<T>(g00, g01, x0);
//		T gx1 = interp<T>(g10, g11, x0);
//		return interp<T>(gx0, gx1, y0);
//	}
//
//	__device__ inline
//	bool ComputeRGB(int& x, int& y, int& u, int& v,
//								   float3& vcg, float3 vcp, float3& vlast, float* row) const {
//
//		float gx = dIx.ptr(v)[u];
//		float gy = dIy.ptr(v)[u];
////		float gx = dIx.ptr(y)[x];
////		float gy = dIy.ptr(y)[x];
//		if(gx * gx + gy * gy < minGxy)
//			return false;
//
//		float3 vl = make_float3(VMapCurr.ptr(y)[x]);
//		if(isnan(vl.x) || vlast.z - vcg.z > 0.7)
//			return false;
//
//		bool valid = GrayCurr.ptr(y)[x] > 0;
//		const int r = 2;
//		for(int i = max(0, u - r ); i < min(u + r, cols - 1); ++i)
//			for(int j = max(0, v - r ); j < min(v + r, rows - 1); ++j)
//				valid = (GrayLast.ptr(j)[i] > 0) && valid;
//
//		if(!valid)
//			return false;
//
//
//
//		float diff = static_cast<float>(GrayCurr.ptr(y)[x]) - static_cast<float>(GrayLast.ptr(v)[u]);
//		float w = abs(diff);
//		w = w > 1e-7 ? 1.0 / w : -1.0 / w;
//		w = 1.0;
//		row[6] = -w * diff;
//		float3 dIdh;
//
//		float invz = 1.0 / vcp.z;
//		dIdh.x = w * gx * fx * invz;
//		dIdh.y = w * gy * fy * invz;
//		dIdh.z = -(dIdh.x * vcp.x + dIdh.y * vcp.y) * invz;
//		*(float3*) &row[0] = -dIdh;
//		*(float3*) &row[3] = cross(dIdh, vcp);
//
//		return true;
//	}
//
//	template<bool bICPOnly> __device__ inline
//	void GetRow(int& i, float* sum) const {
//		int y = i / cols;
//		int x = i - (y * cols);
//		Corresp.ptr(y)[x] = 0;
//
//		int u = 0, v = 0;
//		bool bCorresp = false;
//		bool bRGB = false;
//		bool bView = false;
//		float3 vcurr, vlast, nlast, vcurrp;
//		bCorresp = SearchCorresp(bView, x, y, u, v, vcurr, vlast, nlast, vcurrp);
//		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };
//		float row_rgb[7] = { 0, 0, 0, 0, 0, 0, 0 };
//
//        if(bView && !bICPOnly) {
//        	bRGB = ComputeRGB(x, y, u, v, vcurr, vcurrp, vlast, &row_rgb[0]);
//        	if(bRGB)
//        		Corresp.ptr(y)[x] = 255;
//        }
//
//		if(bCorresp) {
//			nlast = invRlast * nlast;
//			vcurr = invRlast * (vcurr - tlast);
//			vlast = invRlast * (vlast - tlast);
//			float3 nvcross = cross(nlast, vcurr);
//            row[0] = -nlast.x;
//            row[1] = -nlast.y;
//            row[2] = -nlast.z;
//            row[3] = nvcross.x;
//            row[4] = nvcross.y;
//            row[5] = nvcross.z;
//            row[6] = -nlast * (vlast - vcurr);
//		}
//
//		int count = 0;
//		if(bICPOnly || !bRGB) {
//			for(int i = 0; i < 7; ++i)
//				for(int j = i; j < 7; ++j)
//					sum[count++] = row[i] * row[j];
////					memset(sum, 0, sizeof(float)* 29);
////					sum[count++] = row_rgb[i] * row_rgb[j];
//		}
//		else {
//			for(int i = 0; i < 7; ++i)
//				for(int j = i; j < 7; ++j)
////					sum[count++] = icpW * row[i] * row[j] + (1 - icpW) * row_rgb[i] * row_rgb[j];
//					sum[count++] = row_rgb[i] * row_rgb[j];
////					sum[count++] = row[i] * row[j];
//		}
//
//		sum[count] = (float)(bCorresp);
//	}
//
//	template<typename T, int size, bool bRGB>
//	__device__ void operator()() const {
//		T sum[size];
//		T val[size];
//		memset(sum, 0, sizeof(T) * size);
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		for (; i < N; i += blockDim.x * gridDim.x) {
//			memset(val, 0, sizeof(T) * size);
//			GetRow<bRGB>(i, val);
//
//#pragma unroll
//			for(int j = 0; j < size; ++j)
//				sum[j] += val[j];
//		}
//
//		BlockReduceSum<T, size>(sum);
//
//		if (threadIdx.x == 0)
//#pragma unroll
//			for(int i = 0; i < size; ++i)
//				out.ptr(blockIdx.x)[i] = sum[i];
//	}
//};
//
//template<bool bICPOnly> __global__ void
//ICPReduceSum_device(const ICPReduce icp) {
//	icp.template operator()<float, 29, bICPOnly>();
//}
//
//static void inline
//CreateMatrix(float* host_data, float* host_a, float* host_b) {
//    int shift = 0;
//	for (int i = 0; i < 6; ++i)
//		for (int j = i; j < 7; ++j) {
//			float value = host_data[shift++];
//			if (j == 6)
//				host_b[i] = value;
//			else
//				host_a[j * 6 + i] = host_a[i * 6 + j] = value;
//		}
//}
//
//void ICPReduceSum(Frame& NextFrame, Frame& LastFrame, int pyr,
//				  	  	  	  	   float* host_a, float* host_b, float& cost) {
//
//	DeviceArray2D<uchar> Corresp(Frame::cols(pyr), Frame::rows(pyr));
//	DeviceArray2D<float> sum(29, 96);
//	DeviceArray<float> result(29);
//	Corresp.zero();
//	result.zero();
//	sum.zero();
//
//	float minGxy[Frame::numPyrs] = { 25.0, 9.0, 1.0 };
////	float minGxy[Frame::numPyrs] = { 25.0 / 2, 9.0  / 2, 1.0  / 2 };
//
//	ICPReduce icp;
//	icp.out = sum;
//	icp.minGxy = minGxy[pyr];
//	icp.Corresp = Corresp;
//	icp.dIx = LastFrame.mdIx[pyr];
//	icp.dIy = LastFrame.mdIy[pyr];
////	icp.dIx = NextFrame.mdIx[pyr];
////	icp.dIy = NextFrame.mdIy[pyr];
//	icp.VMapCurr = NextFrame.mVMap[pyr];
//	icp.NMapCurr = NextFrame.mNMap[pyr];
//	icp.GrayCurr = NextFrame.mGray[pyr];
//	icp.VMapLast = LastFrame.mVMap[pyr];
//	icp.NMapLast = LastFrame.mNMap[pyr];
//	icp.GrayLast = LastFrame.mGray[pyr];
//	icp.cols = Frame::cols(pyr);
//	icp.rows = Frame::rows(pyr);
//	icp.N = Frame::pixels(pyr);
//	icp.Rcurr = NextFrame.mRcw;
//	icp.tcurr = Converter::CvMatToFloat3(NextFrame.mtcw);
//	icp.Rlast = LastFrame.mRcw;
//	icp.invRlast = LastFrame.mRwc;
//	icp.tlast = Converter::CvMatToFloat3(LastFrame.mtcw);
//	icp.angleThresh = 0.6;
//	icp.distThresh = 0.1;
//	icp.icpW = 0.9;
//	icp.fx = Frame::fx(pyr);
//	icp.fy = Frame::fy(pyr);
//	icp.cx = Frame::cx(pyr);
//	icp.cy = Frame::cy(pyr);
//
//	ICPReduceSum_device<false><<<96, 224>>>(icp);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	ReduceSum<float, 29><<<1, MaxThread>>>(sum, result, 96);
//
//	SafeCall(cudaDeviceSynchronize());
//	SafeCall(cudaGetLastError());
//
//	cv::Mat corr(Frame::rows(pyr), Frame::cols(pyr), CV_8UC1);
//	Corresp.download((void*)corr.data, corr.step);
//	cv::imshow("Corresp", corr);
//
//	float host_data[29];
//	result.download(host_data);
//	CreateMatrix(host_data, host_a, host_b);
//	cost = sqrt(host_data[27]) / host_data[28];
//}

#include "reduction.h"

template<int rows, int cols>
void inline createMatrix(float* host_data, double* host_a,
		double* host_b) {
	int shift = 0;
	for (int i = 0; i < rows; ++i)
		for (int j = i; j < cols; ++j) {
			double value = (double)host_data[shift++];
			if (j == rows)
				host_b[i] = value;
			else
				host_a[j * rows + i] = host_a[i * rows + j] = value;
		}
}

template<typename T, int size> __device__
inline void warpReduce(T* val) {
	#pragma unroll
	for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
		#pragma unroll
		for (int i = 0; i < size; ++i) {
			val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
		}
	}
}

template<typename T, int size> __device__
inline void blockReduce(T* val) {
	static __shared__ T shared[32 * size];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	warpReduce<T, size>(val);

	if (lane == 0)
		memcpy(&shared[wid * size], val, sizeof(T) * size);

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(val, &shared[lane * size], sizeof(T) * size);
	else
		memset(val, 0, sizeof(T) * size);

	if (wid == 0)
		warpReduce<T, size>(val);
}

template<typename T, int size> __global__
void reduce(PtrStep<T> in, T* out, int N) {
	T sum[size];
	memset(sum, 0, sizeof(T) * size);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < N; i += blockDim.x * gridDim.x)
		#pragma unroll
		for (int j = 0; j < size; ++j)
			sum[j] += in.ptr(i)[j];

	blockReduce<T, size>(sum);

	if (threadIdx.x == 0)
		#pragma unroll
		for (int i = 0; i < size; ++i)
			out[i] = sum[i];
}

struct ICPReduce {

	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f invRlast;
	float3 tcurr;
	float3 tlast;
	PtrStep<float4> VMapCurr, VMapLast;
	PtrStep<float4> NMapCurr, NMapLast;
	int cols, rows, N;
	float fx, fy, cx, cy;
	float angleThresh, distThresh;

	mutable PtrStepSz<float> out;

	__device__ inline
	bool searchPoint(int& x, int& y, float3& vcurr_g, float3& vlast_g,
			float3& nlast_g) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if (isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = invRlast * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		int u = (int) (vcurr_p.x * invz * fx + cx + 0.5);
		int v = (int) (vcurr_p.y * invz * fy + cy + 0.5);
		if (u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		float3 vlast_c = make_float3(VMapLast.ptr(v)[u]);
		vlast_g = Rlast * vlast_c + tlast;

		float3 ncurr_c = make_float3(NMapCurr.ptr(y)[x]);
		float3 ncurr_g = Rcurr * ncurr_c;

		float3 nlast_c = make_float3(NMapLast.ptr(v)[u]);
		nlast_g = Rlast * nlast_c;

		float dist = norm(vlast_g - vcurr_g);
		float sine = norm(cross(ncurr_g, nlast_g));

		return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_c.x)
				&& !isnan(nlast_c.x));
	}

	__device__ inline
	void getRow(int& i, float* sum) const {
		int y = i / cols;
		int x = i - (y * cols);

		bool found = false;
		float3 vcurr, vlast, nlast;
		found = searchPoint(x, y, vcurr, vlast, nlast);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if (found) {
			nlast = invRlast * nlast;
			vcurr = invRlast * (vcurr - tlast);
			vlast = invRlast * (vlast - tlast);
			*(float3*) &row[0] = -nlast;
			*(float3*) &row[3] = cross(nlast, vlast);
			row[6] = -nlast * (vlast - vcurr);
		}

		int count = 0;
		#pragma unroll
		for (int i = 0; i < 7; ++i)
			#pragma unroll
			for (int j = i; j < 7; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float) found;
	}

	template<typename T, int size>
	__device__ void operator()() const {
		T sum[size];
		T val[size];
		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			getRow(i, val);
		    #pragma unroll
			for (int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		blockReduce<T, size>(sum);

		if (threadIdx.x == 0)
			#pragma unroll
			for (int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void icpStepKernel(const ICPReduce icp) {
	icp.template operator()<float, 29>();
}

void icpStep(DeviceArray2D<float4> & nextVMap,
		     DeviceArray2D<float4> & lastVMap,
		     DeviceArray2D<float4> & nextNMap,
		     DeviceArray2D<float4> & lastNMap,
		     Matrix3f Rcurr,
		     float3 tcurr,
		     Matrix3f Rlast,
		     Matrix3f RlastInv,
		     float3 tlast,
		     MatK K,
		     DeviceArray2D<float> & sum,
		     DeviceArray<float> & out,
		     float * residual,
		     double * matrixA_host,
		     double * vectorB_host) {

	int cols = nextVMap.cols();
	int rows = nextVMap.rows();

	ICPReduce icp;
	icp.out = sum;
	icp.VMapCurr = nextVMap;
	icp.NMapCurr = nextNMap;
	icp.VMapLast = lastVMap;
	icp.NMapLast = lastNMap;
	icp.cols = cols;
	icp.rows = rows;
	icp.N = cols * rows;
	icp.Rcurr = Rcurr;
	icp.tcurr = tcurr;
	icp.Rlast = Rlast;
	icp.invRlast = RlastInv;
	icp.tlast = tlast;
	icp.angleThresh = 0.6;
	icp.distThresh = 0.1;
	icp.fx = K.fx;
	icp.fy = K.fy;
	icp.cx = K.cx;
	icp.cy = K.cy;

	icpStepKernel<<<96, 224>>>(icp);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	reduce<float, 29> <<<1, MaxThread>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	out.download((void*) host_data);
	createMatrix<6, 7>(host_data, matrixA_host, vectorB_host);

	residual[0] = host_data[27];
	residual[1] = host_data[28];
}

struct SO3Reduction {

	int cols, rows;
	int N;
	float fx, fy, cx, cy;
	Matrix3f homography;
	PtrStep<unsigned char> nextImage;
	PtrStep<unsigned char> lastImage;
	mutable PtrStepSz<float> out;
	Matrix3f kinv, krlr;

	__device__ inline float2 getGradient(const PtrStep<unsigned char> & img,
			int & x, int & y) const {

		float2 gradient;
		float actu = static_cast<float>(img.ptr(y)[x]);

		float back = static_cast<float>(img.ptr(y)[x - 1]);
		float fore = static_cast<float>(img.ptr(y)[x + 1]);
		gradient.x = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

		back = static_cast<float>(img.ptr(y - 1)[x]);
		fore = static_cast<float>(img.ptr(y + 1)[x]);
		gradient.y = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

		return gradient;
	}

	template<typename T>
	__device__ inline void getRow(int & k, T * sum) const {

		int y = k / cols;
		int x = k - (y * cols);
		bool found_coresp = false;

		float3 unwarpedReferencePoint = { (float)x, (float)y, 1.0f };
		float3 warpedReferencePoint = homography * unwarpedReferencePoint;
		int2 warpedReferencePixel = {
				__float2int_rn(warpedReferencePoint.x / warpedReferencePoint.z),
				__float2int_rn(warpedReferencePoint.y / warpedReferencePoint.z)
		};

		if (warpedReferencePixel.x >= 1 &&
			warpedReferencePixel.x < cols - 1 &&
			warpedReferencePixel.y >= 1 &&
			warpedReferencePixel.y < rows - 1 &&
			x >= 1 && x < cols - 1 &&
			y >= 1 && y < rows - 1) {
			found_coresp = true;
		}

		float row[4] = { 0.f, 0.f, 0.f, 0.f };
		if(found_coresp) {
			float2 gradNext = getGradient(
					nextImage,
					warpedReferencePixel.x,
					warpedReferencePixel.y
			);
			float2 gradLast = getGradient(lastImage, x, y);
            float gx = (gradNext.x + gradLast.x) / 2.0f;
            float gy = (gradNext.y + gradLast.y) / 2.0f;
            float3 point = kinv * unwarpedReferencePoint;
            float z2 = point.z * point.z;

			float a = krlr.rowx.x;
			float b = krlr.rowx.y;
			float c = krlr.rowx.z;

			float d = krlr.rowy.x;
			float e = krlr.rowy.y;
			float f = krlr.rowy.z;

			float g = krlr.rowz.x;
			float h = krlr.rowz.y;
			float i = krlr.rowz.z;

			float3 leftProduct = {
					((point.z * (d * gy + a * gx)) - (gy * g * y) - (gx * g * x)) / z2,
			        ((point.z * (e * gy + b * gx)) - (gy * h * y) - (gx * h * x)) / z2,
			      	((point.z * (f * gy + c * gx)) - (gy * i * y) - (gx * i * x)) / z2
			};
			float3 jacRow = cross(leftProduct, point);
			row[0] = jacRow.x;
			row[1] = jacRow.y;
			row[2] = jacRow.z;
			row[3] = -(static_cast<float>(nextImage.ptr(warpedReferencePixel.y)[warpedReferencePixel.x]) - static_cast<float>(lastImage.ptr(y)[x]));
		}

		int count = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
		#pragma unroll
			for (int j = i; j < 4; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float) found_coresp;
	}

	template<typename T, int size>
	__device__ inline void operator()() const {
		T sum[size];
		T val[size];

		memset(sum, 0, sizeof(T) * size);
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			memset(val, 0, sizeof(T) * size);
			getRow(i, val);
			#pragma unroll
			for (int j = 0; j < size; ++j)
				sum[j] += val[j];
		}

		blockReduce<T, size>(sum);

		if (threadIdx.x == 0)
			#pragma unroll
			for (int i = 0; i < size; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void so3StepKernel(const SO3Reduction so3) {
	so3.template operator()<float, 11>();
}

void so3Step(const DeviceArray2D<unsigned char> & nextImage,
			 const DeviceArray2D<unsigned char> & lastImage,
			 Matrix3f homography,
			 Matrix3f kinv,
			 Matrix3f krlr,
			 DeviceArray2D<float> & sum,
			 DeviceArray<float> & out,
			 float * redisual,
			 double * matrixA_host,
			 double * vectorB_host) {

	int cols = nextImage.cols();
	int rows = nextImage.rows();

	SO3Reduction so3;
	so3.nextImage = nextImage;
	so3.lastImage = lastImage;
	so3.homography = homography;
	so3.cols = cols;
	so3.rows = rows;
	so3.N = cols * rows;
	so3.out = sum;

	so3StepKernel<<<96, 224>>>(so3);

	reduce<float, 11><<<1, 1024>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[11];
	out.download((void*) host_data);
	createMatrix<3, 4>(host_data, matrixA_host, vectorB_host);
	redisual[0] = host_data[9];
	redisual[1] = host_data[10];
}

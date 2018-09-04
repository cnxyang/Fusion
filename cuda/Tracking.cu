#include "Tracking.hpp"

__device__ __inline__ JtJJtrSE3 WarpReduceSum(JtJJtrSE3 val) {
#pragma unroll
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.aa += __shfl_down_sync(0xffffffff, val.aa, offset);
        val.ab += __shfl_down_sync(0xffffffff, val.ab, offset);
        val.ac += __shfl_down_sync(0xffffffff, val.ac, offset);
        val.ad += __shfl_down_sync(0xffffffff, val.ad, offset);
        val.ae += __shfl_down_sync(0xffffffff, val.ae, offset);
        val.af += __shfl_down_sync(0xffffffff, val.af, offset);
        val.ag += __shfl_down_sync(0xffffffff, val.ag, offset);

        val.bb += __shfl_down_sync(0xffffffff, val.bb, offset);
        val.bc += __shfl_down_sync(0xffffffff, val.bc, offset);
        val.bd += __shfl_down_sync(0xffffffff, val.bd, offset);
        val.be += __shfl_down_sync(0xffffffff, val.be, offset);
        val.bf += __shfl_down_sync(0xffffffff, val.bf, offset);
        val.bg += __shfl_down_sync(0xffffffff, val.bg, offset);

        val.cc += __shfl_down_sync(0xffffffff, val.cc, offset);
        val.cd += __shfl_down_sync(0xffffffff, val.cd, offset);
        val.ce += __shfl_down_sync(0xffffffff, val.ce, offset);
        val.cf += __shfl_down_sync(0xffffffff, val.cf, offset);
        val.cg += __shfl_down_sync(0xffffffff, val.cg, offset);

        val.dd += __shfl_down_sync(0xffffffff, val.dd, offset);
        val.de += __shfl_down_sync(0xffffffff, val.de, offset);
        val.df += __shfl_down_sync(0xffffffff, val.df, offset);
        val.dg += __shfl_down_sync(0xffffffff, val.dg, offset);

        val.ee += __shfl_down_sync(0xffffffff, val.ee, offset);
        val.ef += __shfl_down_sync(0xffffffff, val.ef, offset);
        val.eg += __shfl_down_sync(0xffffffff, val.eg, offset);

        val.ff += __shfl_down_sync(0xffffffff, val.ff, offset);
        val.fg += __shfl_down_sync(0xffffffff, val.fg, offset);

        val.residual += __shfl_down_sync(0xffffffff, val.residual, offset);
        val.inliers += __shfl_down_sync(0xffffffff, val.inliers, offset);
    }
    return val;
}

__device__ __inline__ JtJJtrSE3 BlockReduceSum(JtJJtrSE3 val) {

    static __shared__ JtJJtrSE3 shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = WarpReduceSum(val);

	if (lane == 0) {
		shared[wid] = val;
	}
    __syncthreads();

    const JtJJtrSE3 zero = { 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0};

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
	if (wid == 0) {
		val = WarpReduceSum(val);
	}

    return val;
}

__global__ void ReduceSum(JtJJtrSE3 * in, JtJJtrSE3 * out, int N) {

    JtJJtrSE3 sum = { 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0};

    int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < N; i += blockDim.x * gridDim.x) {
        sum.add(in[i]);
    }

    sum = BlockReduceSum(sum);

	if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction {

	PtrStep<float4> nextVMap;
	PtrStep<float4> lastVMap;
	PtrStep<float3> nextNMap;
	PtrStep<float3> lastNMap;

	Matrix3f nextR;
	Matrix3f lastR;
	Matrix3f lastRinv;
	float3 nextt;
	float3 lastt;

	float angleThresh;
	float distThresh;
	int cols;
	int rows;
	int N;
	MatK K;

	JtJJtrSE3 * out;

	__device__ inline bool findCorresp(int & x, int & y,
								       float3 & vcurr_g,
								       float3 & vlast_g,
								       float3 & nlast_g) const {

		float3 vcurr_c = make_float3(nextVMap.ptr(y)[x]);
		if (isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = nextR * vcurr_c + nextt;
		float3 vcurr_p = lastRinv * (vcurr_g - lastt);

		float invz = 1.0 / vcurr_p.z;
		int u = (int) (vcurr_p.x * invz * K.fx + K.cx + 0.5);
		int v = (int) (vcurr_p.y * invz * K.fy + K.cy + 0.5);

		if (u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		float3 vlast_c = make_float3(lastVMap.ptr(v)[u]);
		vlast_g = lastR * vlast_c + lastt;

		float3 ncurr_c = nextNMap.ptr(y)[x];
		float3 ncurr_g = nextR * ncurr_c;

		float3 nlast_c = lastNMap.ptr(v)[u];
		nlast_g = lastR * nlast_c;

		float dist = norm(vlast_g - vcurr_g);
		float sine = norm(cross(ncurr_g, nlast_g));

		return (sine < angleThresh &&
				dist <= distThresh &&
				!isnan(ncurr_c.x) &&
				!isnan(nlast_c.x));
	}

	__device__ __inline__ JtJJtrSE3 getProduct(int & k) const {
		int y = k / cols;
		int x = k - (y * cols);
		float3 vcurr, vlast, nlast;
		bool found_coresp = findCorresp(x, y, vcurr, vlast, nlast);

		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };
		if (found_coresp) {
			nlast = lastRinv * nlast;
			vcurr = lastRinv * (vcurr - lastt);
			vlast = lastRinv * (vlast - lastt);
			*(float3*) &row[0] = -nlast;
			*(float3*) &row[3] = cross(nlast, vlast);
			row[6] = -nlast * (vlast - vcurr);
		}

        JtJJtrSE3 val = { row[0] * row[0],
                          row[0] * row[1],
                          row[0] * row[2],
                          row[0] * row[3],
                          row[0] * row[4],
                          row[0] * row[5],
                          row[0] * row[6],

                          row[1] * row[1],
                          row[1] * row[2],
                          row[1] * row[3],
                          row[1] * row[4],
                          row[1] * row[5],
                          row[1] * row[6],

                          row[2] * row[2],
                          row[2] * row[3],
                          row[2] * row[4],
                          row[2] * row[5],
                          row[2] * row[6],

                          row[3] * row[3],
                          row[3] * row[4],
                          row[3] * row[5],
                          row[3] * row[6],

                          row[4] * row[4],
                          row[4] * row[5],
                          row[4] * row[6],

                          row[5] * row[5],
                          row[5] * row[6],

                          row[6] * row[6],
                          (float)found_coresp};
        return val;
	}

	__device__ void operator()() const {

	    JtJJtrSE3 sum = { 0, 0, 0, 0, 0, 0, 0, 0,
	                      0, 0, 0, 0, 0, 0, 0, 0,
	                      0, 0, 0, 0, 0, 0, 0, 0,
	                      0, 0, 0, 0, 0};

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			JtJJtrSE3 val = getProduct(i);
			sum.add(val);
		}

		BlockReduceSum(sum);

		if (threadIdx.x == 0)
			out[blockIdx.x] = sum;
	}
};

__global__ void icpStepKernel(const ICPReduction icp) {
	icp();
}

void icpStep(const DeviceArray2D<float4> & nextVMap,
			 const DeviceArray2D<float4> & lastVMap,
			 const DeviceArray2D<float3> & nextNMap,
			 const DeviceArray2D<float3> & lastNMap,
			 DeviceArray<JtJJtrSE3> & sum,
			 DeviceArray<JtJJtrSE3> & out,
			 float * residual,
			 double * JtJ_host,
			 double * Jtr_host,
			 Matrix3f Rcurr,
			 float3 tcurr,
			 Matrix3f Rlast,
			 Matrix3f RlastInv,
			 float3 tlast,
			 MatK K) {

	int cols = nextVMap.cols();
	int rows = nextVMap.rows();

	ICPReduction icp;

	icp.K = K;
	icp.out = sum;
	icp.cols = cols;
	icp.rows = rows;
	icp.N = cols * rows;

	icp.nextVMap = nextVMap;
	icp.lastVMap = lastVMap;
	icp.nextNMap = nextNMap;
	icp.lastNMap = lastNMap;
	icp.nextR = Rcurr;
	icp.nextt = tcurr;
	icp.lastR = Rlast;
	icp.lastRinv = RlastInv;
	icp.lastt = tlast;
	icp.angleThresh = sin(20.f * 3.14159254f / 180.f);
	icp.distThresh = 0.1;

	icpStepKernel<<<96, 224>>>(icp);

	ReduceSum<<<1, 1024>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[32];
	out.download((JtJJtrSE3*) host_data);

	int shift = 0;
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 7; ++j) {
			double value = (double)host_data[shift++];
			if (j == 6)
				Jtr_host[i] = value;
			else
				JtJ_host[j * 6 + i] = JtJ_host[i * 6 + j] = value;
		}
	}

	residual[0] = host_data[27];
	residual[1] = host_data[28];
}

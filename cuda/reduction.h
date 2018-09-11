#ifndef GPU_REDUCTION_H__
#define GPU_REDUCTION_H__

#include "mathlib.h"
#include "cuarray.h"


struct MatK {
	float fx, fy, cx, cy;
	MatK() : fx(0), fy(0), cx(0), cy(0) {}
	MatK(float fx_, float fy_, float cx_, float cy_)
	:	fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
	MatK operator()(int pyr) {
		int i = 1 << pyr;
		return MatK(fx / i, fy / i, cx / i, cy / i);
	}
};

struct JtJJtrSE3 {
    float aa, ab, ac, ad, ae, af, ag,
              bb, bc, bd, be, bf, bg,
                  cc, cd, ce, cf, cg,
                      dd, de, df, dg,
                          ee, ef, eg,
                              ff, fg;

    float residual, inliers;

	__device__ inline void add(const JtJJtrSE3 & a) {
		aa += a.aa;
		ab += a.ab;
		ac += a.ac;
		ad += a.ad;
		ae += a.ae;
		af += a.af;
		ag += a.ag;

		bb += a.bb;
		bc += a.bc;
		bd += a.bd;
		be += a.be;
		bf += a.bf;
		bg += a.bg;

		cc += a.cc;
		cd += a.cd;
		ce += a.ce;
		cf += a.cf;
		cg += a.cg;

		dd += a.dd;
		de += a.de;
		df += a.df;
		dg += a.dg;

		ee += a.ee;
		ef += a.ef;
		eg += a.eg;

		ff += a.ff;
		fg += a.fg;

		residual += a.residual;
		inliers += a.inliers;
    }
};

struct JtJJtrSO3 {
    float aa, ab, ac, ad,
              bb, bc, bd,
                  cc, cd;

	float residual, inliers;

	__device__ inline void add(const JtJJtrSO3 & a) {
		aa += a.aa;
		ab += a.ab;
		ac += a.ac;
		ad += a.ad;

		bb += a.bb;
		bc += a.bc;
		bd += a.bd;

		cc += a.cc;
		cd += a.cd;

		residual += a.residual;
		inliers += a.inliers;
	}
};

void computeVMap(const DeviceArray2D<float> & depth, DeviceArray2D<float4> & vmap,
				 float fx, float fy, float cx, float cy, float depthCutoff = 3.0f);
void computeVMap(const DeviceArray2D<float4> & vmap, const DeviceArray2D<float4> & nmap);
void rgbImageToIntensity(const DeviceArray2D<uchar3> & rgb, DeviceArray2D<unsigned char> & image);
void pyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);
void pyrDownGauss(const DeviceArray2D<unsigned char> & src, DeviceArray2D<unsigned char> & dst);
void bilateralFilter(const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & filteredDepth, float depthScale);

#include "frame.h"
double ICPReduceSum(DeviceArray2D<float4> & nextVMap, DeviceArray2D<float4> & lastVMap,
					DeviceArray2D<float4> & nextNMap, DeviceArray2D<float4> & lastNMap,
					Frame& NextFrame, Frame& LastFrame, int pyr, double* host_a, double* host_b);


#include <opencv.hpp>
void BuildAdjecencyMatrix(cv::cuda::GpuMat& AM,
						  DeviceArray<ORBKey>& TrainKeys,
						  DeviceArray<ORBKey>& QueryKeys,
						  DeviceArray<float>& MatchDist,
						  DeviceArray<ORBKey>& train_select,
						  DeviceArray<ORBKey>& query_select,
						  DeviceArray<int>& QueryIdx,
						  DeviceArray<int>& SelectedIdx);
#define WarpSize 32
#define MaxThread 1024
#endif

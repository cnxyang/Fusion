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

void icpStep(const DeviceArray2D<float4> & nextVMap,
			 const DeviceArray2D<float4> & lastVMap,
			 const DeviceArray2D<float3> & nextNMap,
			 const DeviceArray2D<float3> & lastNMap,
			 DeviceArray<JtJJtrSE3> & sum,
			 DeviceArray<JtJJtrSE3> & out,
			 float * residual,
			 double * matA,
			 double * vecb,
			 Matrix3f rcurr,
			 float3 tcurr,
			 Matrix3f rlast,
			 Matrix3f rInvlast,
			 float3 tlast,
			 MatK K);

#endif

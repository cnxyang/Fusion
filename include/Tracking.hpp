#ifndef TRACKING_HPP__
#define TRACKING_HPP__

#include "device_map.hpp"
#include "Mapping.hpp"
#include "Viewer.hpp"
#include "Frame.hpp"
#include <vector>

class Viewer;
class Mapping;

using namespace cv;

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

struct MatK {
	float fx, fy, cx, cy;
	MatK() :
			fx(0), fy(0), cx(0), cy(0) {
	}

	MatK(float fx_, float fy_, float cx_, float cy_) :
			fx(fx_), fy(fy_), cx(cx_), cy(cy_) {
	}

	MatK operator()(int pyr) {
		int i = 1 << pyr;
		return MatK(fx / i, fy / i, cx / i, cy / i);
	}
};

void icpStep(const DeviceArray2D<float4> & nextVertex,
			 const DeviceArray2D<float4> & lastVertex,
			 const DeviceArray2D<float3> & nextNormal,
			 const DeviceArray2D<float3> & lastNormal,
			 DeviceArray<JtJJtrSE3> & sum,
			 DeviceArray<JtJJtrSE3> & out,
			 float * residual,
			 double * matA_data,
			 double * vecB_data,
			 MatK K,
			 Frame * nextFrame,
			 Frame * lastFrame);

class Tracking {
public:
	Tracking();
	Tracking(int w, int h, float fx, float fy, float cx, float cy);
	void SetMap(Mapping* pMap);
	void SetViewer(Viewer* pViewer);
	bool Track(Mat& imRGB, Mat& imD);
	void ResetTracking();
//	void AddObservation(const Rendering& render);

public:

	enum State {
		NOT_INITIALISED,
		OK,
		LOST
	};

	bool TrackMap(bool bUseGraph = true);
//	bool TrackICP();

	bool TrackFrame();
	bool InitTracking();
//	void UpdateMap();
//	void UpdateFrame();
	void SetState(State s);
	bool TrackLastFrame();
//	void ShowResiduals();

	Frame mLastFrame;
	Frame mNextFrame;
	State mNextState;
	State mLastState;
	Mapping* mpMap;
	Viewer* mpViewer;

	cv::Mat desc;
	uint mnMapPoints;
	bool mbGraphMatching;
	int mnNoAttempts;
	const float mRotThresh = 0.2;
	const float mTransThresh = 0.05;
	DeviceArray<ORBKey> mDeviceKeys;
	std::vector<ORBKey> mHostKeys;
	std::vector<Eigen::Vector3d> mMapPoints;
	Ptr<cuda::DescriptorMatcher> mORBMatcher;

public:

	void computeICP();
	bool computeSE3();
	void swapFrame();
	bool grabFrame(cv::Mat & imRGB, cv::Mat & imD);
	void needKeyFrame();
	void createKeyFrame(const Frame * f);
	bool trackKeyFrame();

	static const int NUM_PYRS = 3;
	MatK K;
	DeviceArray2D<unsigned short> depth;
	DeviceArray2D<uchar3> color;

	DeviceArray2D<float> lastDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> lastImage[NUM_PYRS];
	DeviceArray2D<float4> lastVMap[NUM_PYRS];
	DeviceArray2D<float3> lastNMap[NUM_PYRS];

	DeviceArray2D<float> nextDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
	DeviceArray2D<float4> nextVMap[NUM_PYRS];
	DeviceArray2D<float3> nextNMap[NUM_PYRS];
	DeviceArray2D<short> nextIdx[NUM_PYRS];
	DeviceArray2D<short> nextIdy[NUM_PYRS];

	DeviceArray<JtJJtrSE3> sumSE3;
	DeviceArray<JtJJtrSO3> sumSO3;
	DeviceArray<JtJJtrSE3> outSE3;
	DeviceArray<JtJJtrSO3> outSO3;

	Frame * nextFrame;
	Frame * lastFrame;

	bool needNewKF;
	unsigned long relocKF;
	KeyFrame * referenceKF;
	KeyFrame * lastKF;

	Eigen::Matrix4d nextPose;
	Eigen::Matrix4d lastPose;
	Eigen::Matrix4d currentPose;
	Eigen::Matrix4d lastUpdatedPose;

	int iteration[NUM_PYRS];
	float icpResidual[2];
	float lastIcpError;
};

#endif

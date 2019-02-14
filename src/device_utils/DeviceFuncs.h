#pragma once

#include <opencv.hpp>

#include "MapStruct.h"
#include "VectorMath.h"
#include "Intrinsics.h"
#include "DeviceArray.h"

void compute_residual_image(const DeviceArray2D<uchar>& image_curr, const DeviceArray2D<uchar>& image_last);

void compute_residual_transformed(const DeviceArray2D<float4>& vmap_curr,
		const DeviceArray2D<float4>& vmap_last,
		const DeviceArray2D<float4>& nmap_last,
		const DeviceArray2D<float>& image_curr,
		const DeviceArray2D<float>& image_last, float* K, Matrix3f r, float3 t);
void compute_residual_transformed_gt(const DeviceArray2D<float4>& vmap_curr,
		const DeviceArray2D<float4>& vmap_last,
		const DeviceArray2D<float4>& nmap_last,
		const DeviceArray2D<float>& image_curr,
		const DeviceArray2D<float>& image_last, float* K, Matrix3f r_gt,
		float3 t_gt, Matrix3f r, float3 t);

void ResetMap(MapStruct map);

void ResetKeyPoints(KeyMap map);

void InsertKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<int> & keyIndex, size_t size);

void CollectKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<uint> & noKeys);

void Raycast(MapStruct map, DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, float invfx, float invfy, float cx, float cy);

bool CreateRenderingBlocks(MapStruct* map,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		const float & depthMax, const float & depthMin,
		DeviceArray<RenderingBlock> & renderingBlockList,
		DeviceArray<uint> & noRenderingBlocks,
		Matrix3f RviewInv, float3 tview,
		uint noVisibleBlocks, float fx, float fy, float cx, float cy);

uint MeshScene(DeviceArray<uint> & noOccupiedBlocks,
		DeviceArray<uint> & noTotalTriangles,
		MapStruct map,
		const DeviceArray<int> & edgeTable,
		const DeviceArray<int> & vertexTable,
		const DeviceArray2D<int> & triangleTable,
		DeviceArray<float3> & normal,
		DeviceArray<float3> & vertex,
		DeviceArray<uchar3> & color,
		DeviceArray<int3> & blockPoses);

void CheckBlockVisibility(MapStruct map, DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv, float3 tview, int cols, int rows,
		float fx, float fy, float cx, float cy, float depthMax, float depthMin,
		uint * host_data);

void FuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & nmap,
		const DeviceArray2D<int>& weight,
		DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, MapStruct map,
		float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

void DeFuseMap(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, const DeviceArray2D<float4> & nmap,
		const DeviceArray2D<int>& weight, DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv, float3 tview, MapStruct map,
		float fx, float fy, float cx, float cy, float depthMax, float depthMin,
		uint * host_data);

void DefuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, const DeviceArray2D<float4> & nmap,
		DeviceArray<uint> & noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, MapStruct map, float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

void FilterDepth(const DeviceArray2D<unsigned short> & depth,
		DeviceArray2D<float> & rawDepth, DeviceArray2D<float> & filteredDepth,
		float depthScale, float depthCutoff);

void ComputeVMap(const DeviceArray2D<float> & depth,
		DeviceArray2D<float4> & vmap, float fx, float fy, float cx, float cy,
		float depthCutoff);

void ComputeNMap(const DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap);

void pyrdown_image_mean(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);

void PyrDownGauss(const DeviceArray2D<unsigned char> & src,
		DeviceArray2D<unsigned char> & dst);

void ImageToIntensity(const DeviceArray2D<uchar3> & rgb,
		DeviceArray2D<unsigned char> & image);

void compute_image_derivatives(const DeviceArray2D<float>& image, DeviceArray2D<float>& dx, DeviceArray2D<float>& dy);

void ResizeMap(const DeviceArray2D<float4> & vsrc,
		const DeviceArray2D<float4> & nsrc, DeviceArray2D<float4> & vdst,
		DeviceArray2D<float4> & ndst);

void RenderImage(const DeviceArray2D<float4>& points,
		const DeviceArray2D<float4>& normals, const float3 light_pose,
		DeviceArray2D<uchar4>& image);

void DepthToImage(const DeviceArray2D<float> & depth,
		DeviceArray2D<uchar4> & image);

void RgbImageToRgba(const DeviceArray2D<uchar3> & image,
		DeviceArray2D<uchar4> & rgba);

void ForwardWarping(const DeviceArray2D<float4> & srcVMap,
		const DeviceArray2D<float4> & srcNMap, DeviceArray2D<float4> & dstVMap,
		DeviceArray2D<float4> & dstNMap, Matrix3f srcRot, Matrix3f dstInvRot,
		float3 srcTrans, float3 dstTrans, float fx, float fy, float cx,
		float cy);

/*
 * Estimate the SO3 transformation of two frames
 * TODO: suspect of compromised performance
 */
void SO3Step(const DeviceArray2D<unsigned char>& nextImage,
			 const DeviceArray2D<unsigned char>& lastImage,
			 const DeviceArray2D<short>& dIdx,
			 const DeviceArray2D<short> & dIdy,
			 Matrix3f RcurrInv, Matrix3f Rlast,
			 CameraIntrinsics K,
			 DeviceArray2D<float>& sum,
			 DeviceArray<float>& out,
			 float * residual,
			 double * matrixA_host,
			 double * vectorB_host);

// Estimate the SE3 transformation between two frames
// this is purely relied on geometry information
// hence coloured images are ignored.
void ICPStep(DeviceArray2D<float4>& nextVMap,
			 DeviceArray2D<float>& nextDepth,
			 DeviceArray2D<float>& lastDepth,
			 DeviceArray2D<float>& dZdx,
			 DeviceArray2D<float>& dZdy,
			 DeviceArray2D<float>& sum,
			 DeviceArray<float>& out,
			 Matrix3f R, float3 t,
			 float* K, float * residual,
			 double * matrixA_host,
			 double * vectorB_host);

// Estimate SE3 transform based on
// direct image alignment
// using depth image only as a cue
void RGBStep(const DeviceArray2D<unsigned char>& nextImage,
			 const DeviceArray2D<unsigned char>& lastImage,
			 const DeviceArray2D<float4>& nextVMap,
			 const DeviceArray2D<float4>& lastVMap,
			 const DeviceArray2D<short>& dIdx,
			 const DeviceArray2D<short>& dIdy,
			 Matrix3f Rcurr, Matrix3f RcurrInv,
			 Matrix3f Rlast, Matrix3f RlastInv,
			 float3 tcurr, float3 tlast, float* K,
			 DeviceArray2D<float> & sum,
			 DeviceArray<float> & out,
			 DeviceArray2D<int> & sumRes,
			 DeviceArray<int> & outRes,
			 float * residual,
			 double * matrixA_host,
			 double * vectorB_host);

struct CorrespItem
{
	bool valid;
	int u, v;
	float icp_residual;
	float rgb_residual;
};
struct ResidualVector
{
	bool valid;
	float icp, rgb;
};

struct Corresp
{
	bool valid;
	int u, v;
};

#include <Eigen/Core>

void initialize_weight(DeviceArray<float>& weight);

void compute_residual_sum(DeviceArray2D<float4>& vmap_curr,
		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& nmap_curr,
		DeviceArray2D<float4>& nmap_last,
		DeviceArray2D<unsigned char>& image_curr,
		DeviceArray2D<unsigned char>& image_last, Matrix3f rcurr, float3 tcurr,
		float* K, DeviceArray2D<float>& sum, DeviceArray<float>& out,
		DeviceArray2D<float> & sumSE3,
		DeviceArray<float> & outSE3, DeviceArray2D<short>& dIdx,
		DeviceArray2D<short>& dIdy, Matrix3f rcurrInv, float * residual,
		double * matrixA_host, double * vectorB_host,
		DeviceArray2D<CorrespItem>& corresp_image);

void FuseKeyFrameDepth(DeviceArray2D<float>& lastDMap,
					   DeviceArray2D<float>& nextDMap,
					   DeviceArray2D<int>& lastWMap,
					   DeviceArray2D<float4>& nextNMap,
					   DeviceArray2D<float4>& nextVMap,
					   Matrix3f R, float3 t,
					   float* K);

void compute_least_square(DeviceArray<Corresp>& corresp,
		DeviceArray<ResidualVector>& residual_vec, DeviceArray<float>& weight,
		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& vmap_curr,
		DeviceArray2D<float4>& nmap_last, DeviceArray2D<float>& dIdx,
		DeviceArray2D<float>& dIdy, DeviceArray2D<float>& sum,
		DeviceArray<float>& out, Matrix3f r, Matrix3f r_inv, float3 t,
		float3 scale, float* intrinsics, double* matrixA_host,
		double* vectorB_host, float* residual);

void compute_weight(DeviceArray<ResidualVector>& residual,
		DeviceArray<float>& weight, float3 scale);

Eigen::Matrix<float, 2, 2> compute_scale(DeviceArray<ResidualVector>& residual,
		DeviceArray<float>& weight, DeviceArray2D<float>& sum,
		DeviceArray<float>& out, int N, float& point_ratio);

void compute_residual(DeviceArray2D<float4>& vmap_curr,
		DeviceArray2D<float4>& vmap_last, DeviceArray2D<float4>& nmap_curr,
		DeviceArray2D<float4>& nmap_last,
		DeviceArray2D<float>& image_curr,
		DeviceArray2D<float>& image_last, DeviceArray<float>& weight,
		DeviceArray<ResidualVector>& residual, DeviceArray<Corresp>& corresp, Matrix3f r,
		float3 t, float* intrinsics);

// ======================= old piece of shit ============================

void BuildAdjacencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & frameKeys,
		DeviceArray<SURF> & mapKeys,
		DeviceArray<float> & dist);

void CheckVisibility(DeviceArray<float3> & pt3d, DeviceArray<float2> & pt2d,
		DeviceArray<int> & match, Matrix3f RcurrInv, float3 tcurr, Matrix3f Rlast,
		float3 tlast, float fx, float fy, float cx, float cy, int cols,
		int rows);

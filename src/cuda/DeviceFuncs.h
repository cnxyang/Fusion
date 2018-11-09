#pragma once

#include <opencv.hpp>

#include "MapStruct.h"
#include "VectorMath.h"
#include "Intrinsics.h"
#include "DeviceArray.h"

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
		DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, MapStruct map,
		float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

void DeFuseMap(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, const DeviceArray2D<float4> & nmap,
		DeviceArray<uint> & noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, MapStruct map, float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

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

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);

void PyrDownGauss(const DeviceArray2D<unsigned char> & src,
		DeviceArray2D<unsigned char> & dst);

void ImageToIntensity(const DeviceArray2D<uchar3> & rgb,
		DeviceArray2D<unsigned char> & image);

void ComputeDerivativeImage(DeviceArray2D<unsigned char> & image,
		DeviceArray2D<short> & dx, DeviceArray2D<short> & dy);

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
			 DeviceArray2D<float4>& nextNMap,
			 DeviceArray2D<float4>& lastVMap,
			 DeviceArray2D<float4>& lastNMap,
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

// ======================= old piece of shit ============================

void BuildAdjacencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & frameKeys,
		DeviceArray<SURF> & mapKeys,
		DeviceArray<float> & dist);

void CheckVisibility(DeviceArray<float3> & pt3d, DeviceArray<float2> & pt2d,
		DeviceArray<int> & match, Matrix3f RcurrInv, float3 tcurr, Matrix3f Rlast,
		float3 tlast, float fx, float fy, float cx, float cy, int cols,
		int rows);

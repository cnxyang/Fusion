#pragma once

#include "VectorMath.h"
#include "Intrinsics.h"
#include <opencv.hpp>
#include "DeviceMap.h"

void ResetMap(DeviceMap map);

void ResetKeyPoints(KeyMap map);

void InsertKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<int> & keyIndex, size_t size);

void CollectKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<uint> & noKeys);

void Raycast(DeviceMap map, DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, float invfx, float invfy, float cx, float cy);

bool CreateRenderingBlocks(const DeviceArray<HashEntry> & visibleBlocks,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		const float & depthMax, const float & depthMin,
		DeviceArray<RenderingBlock> & renderingBlockList,
		DeviceArray<uint> & noRenderingBlocks,
		Matrix3f RviewInv, float3 tview,
		uint noVisibleBlocks, float fx, float fy, float cx, float cy);

uint MeshScene(DeviceArray<uint> & noOccupiedBlocks,
		DeviceArray<uint> & noTotalTriangles,
		DeviceMap map,
		const DeviceArray<int> & edgeTable,
		const DeviceArray<int> & vertexTable,
		const DeviceArray2D<int> & triangleTable,
		DeviceArray<float3> & normal,
		DeviceArray<float3> & vertex,
		DeviceArray<uchar3> & color,
		DeviceArray<int3> & blockPoses);

void CheckBlockVisibility(DeviceMap map, DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv, float3 tview, int cols, int rows,
		float fx, float fy, float cx, float cy, float depthMax, float depthMin,
		uint * host_data);

void FuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & nmap,
		DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceMap map,
		float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

void DefuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, const DeviceArray2D<float4> & nmap,
		DeviceArray<uint> & noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceMap map, float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

struct Residual {

	int diff;
	bool valid;
	int2 curr;
	int2 last;
	float3 point;
};

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

void SO3Step(const DeviceArray2D<unsigned char> & nextImage,
		const DeviceArray2D<unsigned char> & lastImage,
		const DeviceArray2D<short> & dIdx, const DeviceArray2D<short> & dIdy,
		Matrix3f RcurrInv, Matrix3f Rlast, Intrinsics K,
		DeviceArray2D<float> & sum, DeviceArray<float> & out, float * residual,
		double * matrixA_host, double * vectorB_host);

void ICPStep(DeviceArray2D<float4> & nextVMap, DeviceArray2D<float4> & lastVMap,
		DeviceArray2D<float4> & nextNMap, DeviceArray2D<float4> & lastNMap,
		Matrix3f Rcurr, float3 tcurr, Matrix3f Rlast, Matrix3f RlastInv,
		float3 tlast, Intrinsics K, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, float * residual, double * matrixA_host,
		double * vectorB_host);

void RGBStep(const DeviceArray2D<unsigned char> & nextImage,
		const DeviceArray2D<unsigned char> & lastImage,
		const DeviceArray2D<float4> & nextVMap,
		const DeviceArray2D<float4> & lastVMap,
		const DeviceArray2D<short> & dIdx, const DeviceArray2D<short> & dIdy,
		Matrix3f Rcurr, Matrix3f RcurrInv, Matrix3f Rlast, Matrix3f RlastInv,
		float3 tcurr, float3 tlast, Intrinsics K, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, DeviceArray2D<int> & sumRes,
		DeviceArray<int> & outRes, float * residual, double * matrixA_host,
		double * vectorB_host);

void BuildAdjacencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & frameKeys,
		DeviceArray<SURF> & mapKeys,
		DeviceArray<float> & dist);

void CheckVisibility(DeviceArray<float3> & pt3d, DeviceArray<float2> & pt2d,
		DeviceArray<int> & match, Matrix3f RcurrInv, float3 tcurr, Matrix3f Rlast,
		float3 tlast, float fx, float fy, float cx, float cy, int cols,
		int rows);

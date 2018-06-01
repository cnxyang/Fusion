#include "Map.h"
#include "DeviceArray.h"
#include "DeviceMath.h"

__device__ MapDesc pDesc;

void Map::UpdateDesc(MapDesc& desc) {
	memcpy(&mDesc, &desc, sizeof(MapDesc));
	SafeCall(cudaMemcpyToSymbol(pDesc, &mDesc, sizeof(MapDesc)));
}

void Map::DownloadDesc() {
	SafeCall(cudaMemcpyFromSymbol(&mDesc, pDesc, sizeof(MapDesc)));
}

struct FrameFusion {
	DeviceMap Map;
	PtrStep<float> DepthMap;
	PtrStep<uchar3> ColourMap;
	PtrStep<float3> NormalMap;

	float fx, fy, cx, cy;
	float MaxDepth, MinDepth;
	int cols, rows;
	Matrix3f Rcurr, invR;
	float3 tcurr;

	__device__ inline
	bool CheckVertexVisibility(const float3& vg) {
		float3 vc = invR * (vg - tcurr);
		if(vc.z < MinDepth || vc.z > MaxDepth)
			return false;

		float u = fx * vc.x / vc.z + cx;
		float v = fy * vc.y / vc.z + cy;

		return !(u < 0 ||  v < 0 || u >= cols || v >= rows);
	}

	__device__ inline
	bool CheckBlockVisibility(const int3& blockPos) {
		float factor = pDesc.voxelSize * pDesc.blockSize;

		float3 vertex = blockPos * factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.z += factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.y += factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.x += factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.z -= factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.y -= factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.x -= factor;
		vertex.y += factor;
		if(CheckVertexVisibility(vertex)) return true;

		vertex.x += factor;
		vertex.y -= factor;
		vertex.z += factor;
		if(CheckVertexVisibility(vertex)) return true;

		return false;
	}

	__device__
	void CreateVisibleBlocks() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if(x >= cols || y >= rows)
			 return;

		float depth = DepthMap.ptr(y)[x];
		if(isnan(depth) || depth > MaxDepth || depth < MinDepth)
			return;

		float truncdist = pDesc.voxelSize * 8;
		float minD =min(MaxDepth, depth +  truncdist);
		float maxD = min(MaxDepth, depth - truncdist);
		float invSize = 1.0 / pDesc.voxelSize;
	}
};

__global__ void
FuseNewFrame_device() {

}

void FuseNewFrame(const Frame& frame) {

}

#include "Mapper.hpp"
#include "device_math.hpp"

struct Allocator {

	DMap map;
	float dmax, dmin;
	PtrStepSz<float> depth;
	float fx, fy, cx, cy;
	float invfx, invfy;
	Matrix3f R, invR;
	float3 t;

	DEV_FUNC float3 Unproject(int x, int y, float z) {
		float3 v;
		v.z = z;
		v.x = z * (x - cx) * invfx;
		v.y = z * (y - cy) * invfy;
		return v;
	}

	DEV_FUNC float2 Project(float3& v) {
		float2 uv;
		uv.x = fx * v.x / v.z + cx;
		uv.y = fy * v.y / v.z + cy;
		return uv;
	}

	DEV_FUNC bool CheckVertexVisibility(float3 pos) {
		pos = invR * (pos - t);
		if (pos.z < 1e-6f)
			return false;
		float2 uv = Project(pos);
		return uv.x >= 0 && uv.y >= 0 && uv.x < depth.cols && uv.y < depth.rows
				&& pos.z >= dmin && pos.z <= dmax;
	}

	DEV_FUNC bool CheckBlockVisibility(int3& pos) {

		float scale = BlockSize * VoxelSize;
		float3 corner = pos * scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.z += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.y += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.z -= scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.y -= scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x -= scale;
		corner.y += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x += scale;
		corner.y -= scale;
		if (CheckVertexVisibility(corner))
			return true;
		return false;
	}

	DEV_FUNC void CheckVisibility() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.HashTable.size) {
			if (map.VisibilityList[x] == DMap::DoubleCheck
					&& CheckBlockVisibility(map.HashTable[x].pos))
				map.VisibilityList[x] = DMap::Visible;
			else
				map.VisibilityList[x] = DMap::Invisible;
		}
	}

	DEV_FUNC void CreateAllocList() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x < depth.cols && y < depth.rows) {
			float d = depth.ptr(y)[x];
			if (!isnan(d) && d >= dmin && d <= dmax) {
				float start = min(dmax, d - TruncateDist / 2);
				float end = min(dmax, d + TruncateDist / 2);
				if (start >= end)
					return;
				float3 start_pt = (R * Unproject(x, y, start) + t) / VoxelSize;
				float3 end_pt = (R * Unproject(x, y, end) + t) / VoxelSize;
				float3 dir = end_pt - start_pt;
				float length = norm(dir);
				int nSteps = (int) ceil(2.0 * length);
				dir = dir / (float) (nSteps - 1);
				for (int i = 0; i < nSteps; ++i) {
					map.CreateAllocList(start_pt);
					start_pt += dir;
				}
			}
		}
	}

	DEV_FUNC void AllocateMem() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.HashTable.size) {

		}
	}

	DEV_FUNC void FuseDepth() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (blockIdx.x < map.VisibleEntries.size) {

		}
	}
};

CUDA_KERNEL void CreateAllocListKernel(Allocator& alloc) {
	alloc.CreateAllocList();
}

CUDA_KERNEL void BuildVisibilityList(Allocator& alloc) {
	alloc.CheckVisibility();
}

CUDA_KERNEL void AllocateKernel(Allocator& alloc) {
	alloc.AllocateMem();
}

HOST_FUNC void FuseDepth(DMap& map, DeviceArray2D<float>& depth) {
	Allocator alloc;
	alloc.map = map;
	alloc.depth = depth;

	dim3 block(32, 8);
	dim3 grid(cv::divUp(depth.cols(), block.x),
			cv::divUp(depth.rows(), block.y));

	CreateAllocListKernel<<<grid, block>>>(alloc);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	dim3 block_alloc(MaxThread);
	dim3 grid_alloc(cv::divUp(map.HashTable.size, block_alloc.x));

	AllocateKernel<<<grid_alloc, block_alloc>>>(alloc);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct Observor {
	DMap map;
	float fx, fy, cx, cy;
	int cols, rows;
	Matrix3f R, invR;
	float3 t;

	mutable PtrStep<float4> VMap;
	mutable PtrStep<float3> NMap;
	mutable PtrStep<float> dMax;
	mutable PtrStep<float> dMin;

	DEV_FUNC float2 Project(float3& v) {
		float2 uv;
		uv.x = fx * v.x / v.z + cx;
		uv.y = fy * v.y / v.z + cy;
		return uv;
	}

	DEV_FUNC bool ProjectBlock(const int3& pos) {

		int2 topLeft, bottomRight;
		float2 zRange;
		for (int corner = 0; corner < 8; ++corner) {
			int3 tmp = pos;
			tmp.x += (corner & 1) ? 1 : 0;
			tmp.y += (corner & 2) ? 1 : 0;
			tmp.z += (corner & 4) ? 1 : 0;
			float3 v_g = invR * (tmp * BlockSize * VoxelSize - t);
			if (v_g.z < 1e-3f)
				continue;

			float2 uv = Project(v_g);
			uv.x /= 8;
			uv.y /= 8;
			zRange.x = min(zRange.x, v_g.z);
			zRange.y = max(zRange.y, v_g.z);
			topLeft.x = min(topLeft.x, (int) ceil(uv.x));
			topLeft.y = max(topLeft.y, (int) floor(uv.x));
			bottomRight.x = min(bottomRight.x, (int) ceil(uv.y));
			bottomRight.y = max(bottomRight.y, (int) ceil(uv.y));
		}
	}
};

//class Converter {
//public:
//	HOST_FUNC DEV_FUNC static int3 VoxelPosToBlockPos(float3 pos) {
//		pos = pos / VoxelSize;
//		return make_int3(pos);
//	}
//};

CUDA_KERNEL void BuildVisibleBlockListKernel(PtrSz<int> visibilityList,
		PtrSz<int> visibleBlockList) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x > 0 && x < visibilityList.size) {
		int p = visibilityList[x - 1];
		int c = visibilityList[x];
		if (c - p == 1) {
			visibleBlockList[x - 1] = x;
		}
	}
}

HOST_FUNC void BuildVisibleBlockList(DeviceArray<int>& visibilityList,
		DeviceArray<int>& visibleBlockList) {
	dim3 block(MaxThread);
	dim3 grid(cv::divUp(visibilityList.size(), block.x));

	BuildVisibleBlockListKernel<<<grid, block>>>(visibilityList, visibleBlockList);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

uint DMap::Hash(int3& pos) {
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791))
			% nBuckets;
	if (res < 0)
		res += nBuckets;
	return res;
}

HashEntry DMap::CreateEntry(int3& pos) {
	HashEntry entry;
	entry.ptr = -2;
	int x = atomicSub(EntryPtr, 1);
	if (x >= 0) {
		entry.pos = pos;
		entry.ptr = x * BlockSize3;
		entry.offset = -1;
	}

	return entry;
}

void DMap::CreateAllocList(float3& pos) {
	int3 block = make_int3(pos / VoxelSize);
	uint index = Hash(block);
	uint firstEmptyIndex = -1;
	HashEntry* entry = &HashTable[index];
}

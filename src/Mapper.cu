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

DEV_FUNC float2 ProjectVertex(float3& v_c, float& fx, float& fy, float& cx,
		float& cy) {
	float2 uv;
	uv.x = fx * v_c.x / v_c.z + cx;
	uv.y = fy * v_c.y / v_c.z + cy;
	return uv;
}

DEV_FUNC bool ProjectBlock(int3& pos, Block2D& p, Matrix3f& invR, float3& t,
		int& cols, int& rows, float& fx, float& fy, float& cx, float& cy) {

}

template<int threadBlock>
DEV_FUNC int ComputeOffset(uint val, uint* sum) {

	static_assert(threadBlock % 2 == 0, "ComputeOffset");
	__shared__ uint buffer[threadBlock];
	__shared__ uint blockOffset;

	if (threadIdx.x == 0)
		memset(buffer, 0, sizeof(uint) * threadBlock);
	__syncthreads();

	buffer[threadIdx.x] = val;
	__syncthreads();

	int s1, s2;
	for (s1 = 1, s2 = 1; s1 < threadBlock; s1 <<= 1) {
		s2 |= s1;
		if ((threadIdx.x & s2) == s2)
			buffer[threadIdx.x] += buffer[threadIdx.x - s1];
		__syncthreads();
	}

	for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1) {
		if (threadIdx.x != threadBlock - 1 && (threadIdx.x & s2) == s2)
			buffer[threadIdx.x + s1] += buffer[threadIdx.x];
		__syncthreads();
	}

	if (threadIdx.x == 0 && buffer[threadBlock - 1] > 0)
		blockOffset = atomicAdd(sum, buffer[threadBlock - 1]);
	__syncthreads();

	int offset;
	if (threadIdx.x == 0) {
		if (buffer[threadIdx.x] == 0)
			offset = -1;
		else
			offset = blockOffset;
	} else {
		if (buffer[threadIdx.x] == buffer[threadIdx.x - 1])
			offset = -1;
		else
			offset = blockOffset + buffer[threadIdx.x - 1];
	}

	return offset;
}

DEV_FUNC void CreateRenderingBlock(int& offset, Block2D& p) {

}

CUDA_KERNEL void ProjectVisibleBlockKernel(PtrSz<Block2D> proj_block,
		PtrSz<uint> num_need, PtrSz<int> vis_id, PtrSz<HashEntry> blocks,
		int num_vis, Matrix3f invR, float3 t, int cols, int rows, float fx,
		float fy, float cx, float cy) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < num_vis) {
		int id = vis_id[x];
		HashEntry* e = &blocks[id];
		Block2D p;
		bool valid = false;
		valid = ProjectBlock(e->pos, p, invR, t, cols, rows, fx, fy, cx, cy);
		int requiredNumBlock = 0;
		if (valid) {
			int2 rendering_block = make_int2(
					ceilf((float) (p.lowerRight.x - p.upperLeft.x + 1) / 16),
					ceilf((float) (p.lowerRight.y - p.upperLeft.y + 1) / 16));
			requiredNumBlock = rendering_block.x * rendering_block.y;
			if(*num_need + requiredNumBlock >= MaxRenderingBlocks)
				requiredNumBlock = 0;
		}

		int offset = ComputeOffset<512>(requiredNumBlock, num_need);
		if(!valid || offset == -1)
			return;

		CreateRenderingBlock(offset, p);
	}
}

CUDA_KERNEL void CreateRenderingBlockKernel(PtrSz<int> num_need,
		PtrSz<Block2D> proj_block, PtrSz<Block2D> proj_block_tmp) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < num_need.size) {
		int offset = num_need[x];

	}
}

void ProjectVisibleBlock(DeviceArray<Block2D>& proj_block,
		DeviceArray<int>& num_need, DeviceArray<int>& vis_id,
		DeviceArray<HashEntry>& blocks, int num_vis, Matrix3f invR, float3 t,
		int cols, int rows, float fx, float fy, float cx, float cy) {

}

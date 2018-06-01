#include "Map.h"
#include "DeviceMath.h"

extern __device__ MapDesc pDesc;

struct Render {
	DeviceMap map;
	Matrix3f Rcurr;
	Matrix3f invR;
	float3 tcurr;
	int cols, rows;
	float fx, fy, cx, cy, invfx, invfy;
	float MaxD, MinD;
	float voxelSize;
	int blockSize;
	uint* totalDepthBlock;

	mutable PtrSz<DepthBlock> DepthBlocks;
	mutable PtrStep<float4> VMap;
	mutable PtrStep<float3> NMap;
	mutable PtrStep<float2> DepthMap;

	static const int MinMaxSubsample = 8;
	static const int DepthBlockSize = 16;
	static const int MaxDepthBlocks = 65535 * 4;

	__device__ inline
	bool ProjectBlock(const int3& blockPos,
					  	  	      int2& upperleft,
					  	  	      int2& lowerright,
					  	  	      float2& depth) const {

		upperleft = make_int2(cols, rows) / MinMaxSubsample;
		lowerright = make_int2(0, 0);
		depth = make_float2(MaxD, MinD);
		bool valid = false;

		for(int corner = 0; corner < 8; ++corner) {

			int3 tmp = blockPos;
			tmp.x += (corner & 1) ? 1 : 0;
			tmp.y += (corner & 2) ? 1 : 0;
			tmp.z += (corner & 4) ? 1 : 0;
			float3 pt = tmp * pDesc.blockSize * pDesc.voxelSize;
			pt = invR * (pt - tcurr);

			if(pt.z < MinD || pt.z > MaxD)
				continue;

			float2 pixel;
			pixel.x = (fx * pt.x / pt.z + cx) / MinMaxSubsample;
			pixel.y = (fy * pt.x / pt.z + cy) / MinMaxSubsample;

			if(pixel.x < 0 || pixel.y < 0 || pixel.x >= cols || pixel.y >= rows)
				continue;

			if(upperleft.x > floor(pixel.x)) upperleft.x = (int)floor(pixel.x);
			if(upperleft.y > floor(pixel.y)) upperleft.y = (int)floor(pixel.y);
			if(lowerright.x < ceil(pixel.x)) lowerright.x = (int)ceil(pixel.x);
			if(lowerright.y < ceil(pixel.y)) lowerright.y = (int)ceil(pixel.y);
			if(depth.x > pt.z) depth.x = pt.z;
			if(depth.y < pt.z) depth.y = pt.z;

			if(!valid)
				valid = true;
		}

		return valid;
	}

	__device__ inline
	int ComputeOffset(uint element, uint* sum, int numBlocks)
	{
		__shared__ uint Buffer[blockDim.x];
		__shared__ uint Offset;

		Buffer[threadIdx.x] = element;
		__syncthreads();

		int s1, s2;

		for (s1 = 1, s2 = 1; s1 < blockDim.x; s1 <<= 1) {
			s2 |= s1;
			if ((threadIdx.x & s2) == s2)
				Buffer[threadIdx.x] += Buffer[threadIdx.x - s1];
			__syncthreads();
		}

		for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1) {
			if (threadIdx.x != blockDim.x - 1 && (threadIdx.x & s2) == s2)
				Buffer[threadIdx.x + s1] += Buffer[threadIdx.x];
			__syncthreads();
		}

		blockDim.x = ((blockDim.x <= numBlocks) ? blockDim.x : numBlocks);
		if (threadIdx.x == 0 && Buffer[blockDim.x - 1] > 0)
			Offset = atomicAdd(sum, Buffer[blockDim.x - 1]);
		__syncthreads();

		int offset;
		if (threadIdx.x == 0) {
			if (Buffer[threadIdx.x] == 0)
				offset = -1;
			else
				offset = Offset;
		}
		else {
			if (Buffer[threadIdx.x] == Buffer[threadIdx.x - 1])
				offset = -1;
			else
				offset = Offset + Buffer[threadIdx.x - 1];
		}
		return offset;
	}

	__device__ inline
	void FillDepthBlocks() const {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;

		const HashEntry & entry(map.visibleEntries[idx]);

		if(entry.ptr == EntryAvailable || idx >= num_blocks) return;

		int2 upperLeft;
		int2 lowerRight;
		float2 zRange;

		bool validProjection = false;

		validProjection = ProjectSingleBlock(entry.pos, upperLeft, lowerRight, zRange);

		int2 requiredRenderingBlocks = make_int2(ceilf((float)(lowerRight.x - upperLeft.x + 1) / renderingBlockSizeX),
												 	 	    		      ceilf((float)(lowerRight.y - upperLeft.y + 1) / renderingBlockSizeY));

		size_t requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;
		if (!validProjection) requiredNumBlocks = 0;
		if (*noTotalBlocks + requiredNumBlocks >= MAX_RENDERING_BLOCKS) return;
		int out_offset = ComputeOffset(requiredNumBlocks, &noTotalBlocks, num_blocks);
		if (!validProjection) return;
		if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

		CreateRenderingBlocks(out_offset, upperLeft, lowerRight, zRange);
	}

	__device__
	void RenderDepthMap() {
		int x = threadIdx.x;
		int y = threadIdx.y;
		int blockId = blockIdx.x * 4 + blockIdx.y;

		const DepthBlock& block(DepthBlocks[blockId]);
		int xpos = block.upperLeft.x + x;
		if (xpos > block.lowerRight.x) return;
		int ypos = block.upperLeft.y + y;
		if (ypos > block.lowerRight.y) return;

		float * minData = &DepthMap.ptr(ypos)[xpos].x;
		float * maxData = &DepthMap.ptr(ypos)[xpos].y;

		atomicMin(minData, block.depth.x);
		atomicMax(maxData, block.depth.y);
		return;
	}

	__device__ inline
	uint HashIndex(const int3& blockPos) const {

		return ((blockPos.x * 73856093) ^
				     (blockPos.y * 19349669) ^
				     (blockPos.z * 83492791)) & pDesc.hashMask;
	}

	__device__ inline
	HashEntry FindHashEntry(float3 pt) const {

		int3 blockPos = make_int3(pt);
		if(blockPos.x) blockPos.x -= pDesc.blockSize;
		if(blockPos.y) blockPos.y -= pDesc.blockSize;
		if(blockPos.z) blockPos.z -= pDesc.blockSize;

		uint BucketId = HashIndex(blockPos);
		uint EntryId = BucketId * pDesc.bucketSize;

		for(int i = 0; i < pDesc.bucketSize; ++i, ++EntryId) {
			HashEntry curr = map.hashEntries[EntryId];
			if(curr.pos == blockPos)
				return curr;
		}

		uint LastEntryId = (BucketId + 1) * pDesc.bucketSize - 1;
		EntryId = LastEntryId;

		for(int i = 0; i < pDesc.maxLinkedList; ++i) {
			HashEntry curr = map.hashEntries[EntryId];
			if(curr.pos == blockPos)
				return curr;

			if(curr.offset == 0)
				break;

			EntryId = (LastEntryId + curr.offset) & pDesc.hashMask;
		}

		HashEntry entry;
		entry.ptr = EntryAvailable;
		return entry;
	}

	__device__ inline
	Voxel FindVoxel(float3 pt) const {

		HashEntry curr = FindHashEntry(pt);
		if(curr.ptr == EntryAvailable) {
			Voxel voxel;
			voxel.sdf = 1.0f;
			return voxel;
		}

		return map.voxelBlocks[curr.ptr];
	}

	__device__ inline
	float ReadSdf(float3& pt) const {
		return FindVoxel(pt).sdf;
	}

	__device__ inline
	float ReadSdfInterp(const float3& pt) const {

		float res1, res2, v1, v2;
		float3 coeff;
		coeff = pt - floor(pt);

		v1 = FindVoxel(pt + make_float3(0, 0, 0)).sdf;
		v2 = FindVoxel(pt + make_float3(1, 0, 0)).sdf;
		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = FindVoxel(pt + make_float3(0, 1, 0)).sdf;
		v2 = FindVoxel(pt + make_float3(1, 1, 0)).sdf;
		res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		v1 = FindVoxel(pt + make_float3(0, 0, 1)).sdf;
		v2 = FindVoxel(pt + make_float3(1, 0, 1)).sdf;
		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = FindVoxel(pt + make_float3(0, 1, 1)).sdf;
		v2 = FindVoxel(pt + make_float3(1, 1, 1)).sdf;
		res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		return (1.0f - coeff.z) * res1 + coeff.z * res2;
	}

	__device__ inline
	float ReadSdfInterpConf(float3& pt, float& conf) const {

		float res1, res2, v1, v2;
		float res1_c, res2_c, v1_c, v2_c;

		float3 coeff; Voxel voxel;
		coeff = pt - floor(pt);

		voxel = FindVoxel(pt + make_float3(0, 0, 0)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = FindVoxel(pt + make_float3(1, 0, 0)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
		res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

		voxel = FindVoxel(pt + make_float3(0, 1, 0)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = FindVoxel(pt + make_float3(1, 1, 0)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
		res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

		voxel = FindVoxel(pt + make_float3(0, 0, 1)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = FindVoxel(pt + make_float3(1, 0, 1)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
		res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

		voxel = FindVoxel(pt + make_float3(0, 1, 1)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = FindVoxel(pt + make_float3(1, 1, 1)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
		res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

		conf = (1.0f - coeff.z) * res1_c + coeff.z * res2_c;

		return (1.0f - coeff.z) * res1 + coeff.z * res2;
	}

	__device__ inline
	void TraceRay() const {

		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;

		float3 ptNear, ptFar;
		float3 blockNear, blockFar;
		float scale, confidence;
		bool found = false;

		ptNear.z = DepthMap.ptr(y)[x].x;
		ptNear.x = ptNear.z * (x - cx) * invfx;
		ptNear.y = ptNear.z * (y - cy) * invfy;
		blockNear = (Rcurr * ptNear + tcurr) / pDesc.voxelSize;
		float length = norm(ptNear) / pDesc.voxelSize;

		ptFar.z = DepthMap.ptr(y)[x].y;
		ptFar.x = ptFar.z * (x - cx) * invfx;
		ptFar.y = ptFar.z * (y - cy) * invfy;
		blockFar = (Rcurr * ptFar + tcurr) / pDesc.voxelSize;
		float maxlength = norm(ptFar) / pDesc.voxelSize;

		float3 ray = normalised(blockFar - blockNear);
		float sdf;
		float step = 0;
		while(length < maxlength) {
			HashEntry entry = FindHashEntry(blockNear);
			if(entry.ptr == EntryAvailable) {
				step = pDesc.blockSize;
			}
			else {
				sdf = ReadSdf(blockNear);

				if((sdf <= 0.1f) && (sdf >= -0.5f))
					sdf = ReadSdfInterp(blockNear);

				if(sdf <= 0.0f)
					break;

				step = max(sdf * scale, 1.0f);
			}
			blockNear += step * ray;
			length += step;
		}

		if(sdf <= 0.0f) {
			step = sdf * scale;
			blockNear += step * ray;

			sdf = ReadSdfInterpConf(blockNear, confidence);

			step = sdf * scale;
			blockNear += step * ray;
			found = true;
		}
		else
			found = false;

		if(found) {
			VMap.ptr(y)[x] = make_float4(invR * (blockNear * voxelSize - tcurr), 1.0f);
			VMap.ptr(y)[x].w = confidence + 1.0f;
		}
		else
			VMap.ptr(y)[x].w = 0.0f;
	}
};

__global__ void
ProjectBlocks_device(const Render rd) {
	rd.RenderDepthMap();
}

__global__ void
RenderMap_device(const Render rd) {
	rd.TraceRay();
}

void Map::RenderMap(Rendering& render) {

	Render rd;
	rd.cols = render.cols;
	rd.rows = render.rows;
	rd.MaxD = render.maxD;
	rd.MinD = render.minD;
	rd.cx = render.cx;
	rd.cy = render.cy;
	rd.fx = render.fx;
	rd.fy = render.fy;
	rd.VMap = render.VMap;
	rd.NMap = render.NMap;
}

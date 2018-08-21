#include "Mapping.hpp"
#include "Timer.hpp"
#include "device_array.hpp"
#include "device_math.hpp"
#include "device_mapping.cuh"

#define CUDA_KERNEL __global__

template<int threadBlock>
DEV_FUNC int ComputeOffset(uint element, uint *sum) {

	__shared__ uint buffer[threadBlock];
	__shared__ uint blockOffset;

	if (threadIdx.x == 0)
		memset(buffer, 0, sizeof(uint) * 16 * 16);
	__syncthreads();

	buffer[threadIdx.x] = element;
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

struct HashIntegrator {

	DeviceMap map;
	float invfx, invfy;
	float fx, fy, cx, cy;
	float DEPTH_MIN, DEPTH_MAX;
	int cols, rows;
	Matrix3f Rot;
	Matrix3f invRot;
	float3 trans;

	uint* nVisibleBlock;
	PtrStep<float> depth;
	PtrSz<int> blockVisibilityList;
	PtrSz<HashEntry> visibleBlockList;

	DEV_FUNC float2 ProjectVertex(float3& pt3d) {
		float2 pt2d;
		pt2d.x = fx * pt3d.x / pt3d.z + cx;
		pt2d.y = fy * pt3d.y / pt3d.z + cy;
		return pt2d;
	}

	DEV_FUNC float3 UnprojectWorld(const int& x, const int& y, const float& z) {
		float3 pt3d;
		pt3d.z = z;
		pt3d.x = z * (x - cx) * invfx;
		pt3d.y = z * (y - cy) * invfy;
		return Rot * pt3d + trans;
	}

	DEV_FUNC bool CheckVertexVisibility(float3 pt3d) {
		pt3d = invRot * (pt3d - trans);
		if (pt3d.z < 1e-3f)
			return false;
		float2 pt2d = ProjectVertex(pt3d);

		return pt2d.x >= 0 && pt2d.y >= 0 && pt2d.x < cols && pt2d.y < rows
				&& pt3d.z >= DEPTH_MIN && pt3d.z <= DEPTH_MAX;
	}

	DEV_FUNC bool CheckBlockVisibility(const int3& pos) {

		float scale = BLOCK_DIM * VOXEL_SIZE;
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
		corner.z += scale;
		if (CheckVertexVisibility(corner))
			return true;
		return false;
	}

	DEV_FUNC void CreateBlocks() {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < cols && y < rows) {
			float z = depth.ptr(y)[x];
			if (!isnan(z) && z >= DEPTH_MIN && z <= DEPTH_MAX) {
				float thresh = TRUNC_DIST / 2;
				float z_near = min(DEPTH_MAX, z - thresh);
				float z_far = min(DEPTH_MAX, z + thresh);
				if (z_near >= z_far)
					return;
				float3 pt_near = UnprojectWorld(x, y, z_near) / VOXEL_SIZE;
				float3 pt_far = UnprojectWorld(x, y, z_far) / VOXEL_SIZE;
				float3 dir = pt_far - pt_near;
				float length = norm(dir);
				int nSteps = (int) ceil(2.0 * length);
				dir = dir / (float) (nSteps - 1);
				for (int i = 0; i < nSteps; ++i) {
					int3 blockPos = map.voxelPosToBlockPos(make_int3(pt_near));
					map.CreateBlock(blockPos);
					pt_near += dir;
				}
			}
		}
	}

	DEV_FUNC void CheckFullVisibility() {
		__shared__ bool bScan;
		if (threadIdx.x == 0)
			bScan = false;
		__syncthreads();
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		HashEntry& entry = map.hashEntries[x];
		uint val = 0;
		if (entry.ptr != EntryAvailable) {
			if (CheckBlockVisibility(entry.pos)) {
				bScan = true;
				val = 1;
			}
		}
		__syncthreads();
		if (bScan) {
			int offset = ComputeOffset<1024>(val, map.noVisibleBlocks);
			if(offset != -1)
				map.visibleEntries[offset] = map.hashEntries[x];
		}
	}


	template<bool fuse>
	DEV_FUNC void UpdateMap() {

		HashEntry& entry = map.visibleEntries[blockIdx.x];
		int idx = threadIdx.x;

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);
		int3 voxel_pos = block_pos + map.localIdxToLocalPos(idx);
		float3 pos = map.voxelPosToWorldPos(voxel_pos);

		pos = invRot * (pos - trans);

		float2 pixel = ProjectVertex(pos);

		if (pixel.x < 1 || pixel.y < 1 || pixel.x >= cols - 1
				|| pixel.y >= rows - 1)
			return;

		int2 uv = make_int2(pixel + make_float2(0.5, 0.5));

		float dp_scaled = depth.ptr(uv.y)[uv.x];

		if (isnan(dp_scaled) || dp_scaled > DEPTH_MAX || dp_scaled < DEPTH_MIN)
			return;

		float trunc_dist = TRUNC_DIST;

		float sdf = dp_scaled - pos.z;

		if (sdf >= -trunc_dist) {
			sdf = fmin(1.0f, sdf / trunc_dist);

			Voxel curr;
			curr.sdf = sdf;
//			curr.rgb = colour.ptr(uv.y)[uv.x];
			curr.sdfW = 1;

			Voxel & prev = map.voxelBlocks[entry.ptr + idx];

			if (fuse) {
				if (prev.sdfW < 1e-7)
					curr.sdfW = 1;
				prev += curr;
			} else {
				prev -= curr;
			}
		}
	}
};

template<bool fuse>
CUDA_KERNEL void hashIntegrateKernal(HashIntegrator hi) {
	hi.UpdateMap<fuse>();
}

CUDA_KERNEL void compacitifyEntriesKernel(HashIntegrator hi) {
	hi.CheckFullVisibility();
}

CUDA_KERNEL void createBlocksKernel(HashIntegrator hi) {
	hi.CreateBlocks();
}

int Mapping::FuseFrame(const Frame& frame) {

	int pyr = 0;
	HashIntegrator HI;
	HI.map = *this;
	HI.Rot = frame.Rot_gpu();
	HI.invRot = frame.RotInv_gpu();
	HI.trans = frame.Trans_gpu();
	HI.fx = Frame::fx(pyr);
	HI.fy = Frame::fy(pyr);
	HI.cx = Frame::cx(pyr);
	HI.cy = Frame::cy(pyr);
	HI.invfx = 1.0 / Frame::fx(pyr);
	HI.invfy = 1.0 / Frame::fy(pyr);
	HI.depth = frame.mDepth[pyr];
	HI.cols = Frame::cols(pyr);
	HI.rows = Frame::rows(pyr);
	HI.DEPTH_MAX = DEPTH_MAX;
	HI.DEPTH_MIN = DEPTH_MIN;

	dim3 block(32, 8);
	dim3 grid(cv::divUp(Frame::cols(pyr), block.x),
			cv::divUp(Frame::rows(pyr), block.y));

	Timer::StartTiming("Mapping", "Create Blocks");
	createBlocksKernel<<<grid, block>>>(HI);
	Timer::StopTiming("Mapping", "Create Blocks");

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	dim3 block1(1024);
	dim3 grid1(cv::divUp((int) NUM_ENTRIES, block1.x));

	mNumVisibleEntries.zero();
	Timer::StartTiming("Mapping", "Create Visible List");
	compacitifyEntriesKernel<<<grid1, block1>>>(HI);
	Timer::StopTiming("Mapping", "Create Visible List");

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	uint noblock = 0;
	mNumVisibleEntries.download((void*) &noblock);
	if (noblock == 0)
		return 0;

	dim3 block2(512);
	dim3 grid2(noblock);

	Timer::StartTiming("Mapping", "Integrate Depth");
	hashIntegrateKernal<true> <<<grid2, block2>>>(HI);
	Timer::StopTiming("Mapping", "Integrate Depth");

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	mCamTrace.push_back(frame.mPose.topRightCorner(3, 1));

	return noblock;
}

CUDA_KERNEL void resetHashKernel(HashEntry * hash, HashEntry * compact) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= NUM_ENTRIES)
		return;

	hash[idx].release();
	compact[idx].release();
}

CUDA_KERNEL void resetHeapKernel(int * heap, int * heap_counter,
		Voxel * voxels) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > NUM_SDF_BLOCKS)
		return;

	heap[idx] = NUM_SDF_BLOCKS - idx - 1;

	uint block_idx = idx * BLOCK_SIZE;

	for (uint i = 0; i < BLOCK_SIZE; ++i, ++block_idx) {
		voxels[block_idx].release();
	}

	if (idx == 0)
		*heap_counter = NUM_SDF_BLOCKS - 1;
}

CUDA_KERNEL void resetMutexKernel(int * mutex) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= NUM_BUCKETS)
		return;

	mutex[idx] = EntryAvailable;
}

void Mapping::ResetDeviceMemory() {

	dim3 block(1024);
	dim3 grid;

	grid.x = cv::divUp((int) NUM_SDF_BLOCKS, block.x);
	resetHeapKernel<<<grid, block>>>(mMemory, mUsedMem, mVoxelBlocks);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	grid.x = cv::divUp((int) NUM_ENTRIES, block.x);
	resetHashKernel<<<grid, block>>>(mHashEntries, mVisibleEntries);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	grid.x = cv::divUp((int) NUM_BUCKETS, block.x);
	resetMutexKernel<<<grid, block>>>(mBucketMutex);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	ResetKeys(*this);
	mCamTrace.clear();
}

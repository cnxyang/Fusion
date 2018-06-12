#include "Map.h"
#include "Converter.h"
#include "DeviceArray.h"
#include "DeviceMath.h"

//__device__ MapDesc pDesc;
//
//void Map::UpdateDesc(MapDesc& desc) {
//	memcpy(&mDesc, &desc, sizeof(MapDesc));
//	SafeCall(cudaMemcpyToSymbol(pDesc, &mDesc, sizeof(MapDesc)));
//}
//
//void Map::DownloadDesc() {
//	SafeCall(cudaMemcpyFromSymbol(&mDesc, pDesc, sizeof(MapDesc)));
//}

struct HashIntegrator {
	DeviceMap map;
	PtrStep<float> depth;
	PtrStep<uchar3> colour;
	PtrStep<float3> nmap;

	float fx, fy, cx, cy;
	float DEPTH_MIN, DEPTH_MAX;
	uint width;
	uint height;
	Matrix3f R_curr;
	Matrix3f R_inv;
	float3 t_curr;

	__device__ __forceinline__
	bool CheckVertexVisibility(const float3 & v_g) {

		float3 v_c = R_inv * (v_g - t_curr);
		if (v_c.z < 1e-10f)
			return false;
		float2 pixel = CameraVertexToImageF(v_c);

		return pixel.x >= 0 && pixel.y >= 0 && pixel.x < width
				&& pixel.y < height && v_c.z >= DEPTH_MIN && v_c.z <= DEPTH_MAX;
	}

	__device__ __forceinline__
	bool CheckBlockVisibility(const int3 & blockPos) {

		float factor = (float) BLOCK_DIM * VOXEL_SIZE;

		float3 blockCorner = blockPos * factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.z += factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.y += factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.x += factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.z -= factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.y -= factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.x -= factor;
		blockCorner.y += factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		blockCorner.x += factor;
		blockCorner.y -= factor;
		blockCorner.z += factor;
		if (CheckVertexVisibility(blockCorner))
			return true;

		return false;
	}

	__device__ __forceinline__
	float3 depthToCameraVertex(uint x, uint y, float dp) const {

		return make_float3(dp * (x - cx) / fx,
										 dp * (y - cy) / fy, dp);
	}

	__device__ __forceinline__
	float2 CameraVertexToImageF(float3 pos) {

		return make_float2(fx * pos.x / pos.z + cx,
										 fy * pos.y / pos.z + cy);
	}

	__device__ __forceinline__
	float3 DepthToWorldVertex(uint x, uint y, float dp) {

		float3 v_c = depthToCameraVertex(x, y, dp);
		return R_curr * v_c + t_curr;
	}

	__device__
	HashEntry createHashEntry(const int3 & pos, const int & offset) {
		int old = atomicSub(map.heapCounter, 1);
		int ptr = map.heapMem[old];
		if (ptr != -1)
			return HashEntry(pos, ptr * BLOCK_SIZE, offset);
		return HashEntry(pos, EntryAvailable, offset);
	}

	__device__
	void CreateBlock(const int3 & blockPos) {
		uint bucketId = computeHashPos(blockPos, NUM_BUCKETS);
		uint entryId = bucketId * BUCKET_SIZE;

		int firstEmptySlot = -1;
		for (uint i = 0; i < BUCKET_SIZE; ++i, ++entryId) {

			const HashEntry & curr = map.hashEntries[entryId];
			if (curr.pos == blockPos)
				return;
			if (firstEmptySlot == -1 && curr.ptr == EntryAvailable) {
				firstEmptySlot = entryId;
			}
		}

		const uint lastEntryIdx = (bucketId + 1) * BUCKET_SIZE - 1;
		entryId = lastEntryIdx;
		for (int i = 0; i < LINKED_LIST_SIZE; ++i) {
			HashEntry & curr = map.hashEntries[entryId];
			if (curr.pos == blockPos)
				return;
			if (curr.offset == 0)
				break;

			entryId = lastEntryIdx + curr.offset % (BUCKET_SIZE * NUM_BUCKETS);
		}

		if (firstEmptySlot != -1) {
			int old = atomicExch(&map.bucketMutex[bucketId], EntryOccupied);

			if (old != EntryOccupied) {
				HashEntry & entry = map.hashEntries[firstEmptySlot];
				entry = createHashEntry(blockPos, 0);
				atomicExch(&map.bucketMutex[bucketId], EntryAvailable);
			}

			return;
		}

		uint offset = 0;

		for (int i = 0; i < LINKED_LIST_SIZE; ++i) {
			++offset;

			entryId = (lastEntryIdx + offset) % (BUCKET_SIZE * NUM_BUCKETS);
			if (entryId % BUCKET_SIZE == 0) {
				--i;
				continue;
			}

			HashEntry & curr = map.hashEntries[entryId];

			if (curr.ptr == EntryAvailable) {
				int old = atomicExch(&map.bucketMutex[bucketId], EntryOccupied);
				if (old == EntryOccupied)
					return;
				HashEntry & lastEntry = map.hashEntries[lastEntryIdx];
				uint bucketId2 = entryId / BUCKET_SIZE;

				old = atomicExch(&map.bucketMutex[bucketId2], EntryOccupied);
				if (old == EntryOccupied)
					return;

				curr = createHashEntry(blockPos, lastEntry.offset);
				atomicExch(&map.bucketMutex[bucketId], EntryAvailable);
				atomicExch(&map.bucketMutex[bucketId2], EntryAvailable);
				lastEntry.offset = offset;
				map.hashEntries[entryId] = lastEntry;

				return;
			}
		}
		return;
	}

	__device__
	void CreateVisibleBlocks() {
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= height)
			return;

		float dp_scaled = depth.ptr(y)[x];
		if (isnan(dp_scaled) || dp_scaled > DEPTH_MAX || dp_scaled < DEPTH_MIN)
			return;

		float trunc_dist = TRUNC_DIST / 2;
		float dp_min = min(DEPTH_MAX, dp_scaled - trunc_dist);
		float dp_max = min(DEPTH_MAX, dp_scaled + trunc_dist);

		float oneOverVoxelSize = 1.f / VOXEL_SIZE;

		if (dp_min >= dp_max)
			return;

		float3 point = DepthToWorldVertex(x, y, dp_min) * oneOverVoxelSize;

		float3 point_e = DepthToWorldVertex(x, y, dp_max) * oneOverVoxelSize;

		float3 direction = point_e - point;

		float snorm = sqrt(
				direction.x * direction.x + direction.y * direction.y
						+ direction.z * direction.z);

		int noSteps = (int) ceil(2.0f * snorm);

		direction = direction / (float) (noSteps - 1);

		for (int i = 0; i < noSteps; i++) {
			int3 blockPos = map.voxelPosToBlockPos(make_int3(point));

			CreateBlock(blockPos);

			point += direction;
		}
	}

	__device__
	void createBlock(const float3 & pos) {
		CreateBlock(map.worldPosToBlockPos(pos));
	}

	__device__
	void compactifyEntries() {
		const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= NUM_ENTRIES)
			return;
		HashEntry & entry = map.hashEntries[idx];
		if (entry.ptr == EntryAvailable)
			return;

		if (!CheckBlockVisibility(entry.pos))
			return;

		int old = atomicAdd(map.noVisibleBlocks, 1);
		map.visibleEntries[old] = entry;
	}

	__device__
	void operator()(bool fuse) {
		const HashEntry & entry = map.visibleEntries[blockIdx.x];

		int idx = threadIdx.x;

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);
		int3 voxel_pos = block_pos + map.localIdxToLocalPos(idx);
		float3 pos = map.voxelPosToWorldPos(voxel_pos);

		pos = R_inv * (pos - t_curr);

		float2 pixel = CameraVertexToImageF(pos);

		if (pixel.x < 1 || pixel.y < 1 || pixel.x >= width - 1
				|| pixel.y >= height - 1)
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
				if(prev.sdfW < 1e-7)
					curr.sdfW = 1;
				prev += curr;
			} else {
				prev -= curr;
			}
		}
	}
};

template<bool fuse> __global__
void hashIntegrateKernal(HashIntegrator hi) {
	hi(fuse);
}

__global__
void compacitifyEntriesKernel(HashIntegrator hi) {
	hi.compactifyEntries();
}

__global__
void createBlocksKernel(HashIntegrator hi) {
	hi.CreateVisibleBlocks();
}

int Map::FuseFrame(const Frame& frame) {

		int pyr = 0;
		HashIntegrator HI;
		HI.map = *this;
		HI.R_curr = frame.mRcw;
		HI.R_inv = frame.mRwc;
		HI.t_curr = Converter::CvMatToFloat3(frame.mtcw);
		HI.fx = Frame::fx(pyr);
		HI.fy = Frame::fy(pyr);
		HI.cx = Frame::cx(pyr);
		HI.cy = Frame::cy(pyr);
		HI.depth = frame.mDepth[pyr];
		HI.width = Frame::cols(pyr);
		HI.height = Frame::rows(pyr);
		HI.nmap = frame.mNMap[pyr];
		HI.DEPTH_MAX = DEPTH_MAX;
		HI.DEPTH_MIN = DEPTH_MIN;

	    dim3 block(32, 8);
	    dim3 grid(cv::divUp(Frame::cols(pyr), block.x), cv::divUp(Frame::rows(pyr), block.y));

	    createBlocksKernel<<<grid, block>>>(HI);

	    SafeCall(cudaGetLastError());
	    SafeCall(cudaDeviceSynchronize());

	    dim3 block1(1024);
		dim3 grid1(cv::divUp((int)NUM_ENTRIES, block1.x));

	    mNumVisibleEntries.zero();
	    compacitifyEntriesKernel<<<grid1, block1>>>(HI);

	    SafeCall(cudaGetLastError());
	    SafeCall(cudaDeviceSynchronize());

	    uint noblock = 0;
	    mNumVisibleEntries.download((void*)&noblock);
	    if(noblock == 0)
	    	return 0;

	    dim3 block2(512);
	    dim3 grid2(noblock);

	    hashIntegrateKernal<true><<<grid2, block2>>>(HI);

	    SafeCall(cudaGetLastError());
	    SafeCall(cudaDeviceSynchronize());

	    return noblock;
}

__global__
void resetHashKernel(HashEntry * hash,
				     HashEntry * compact)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= NUM_ENTRIES) return;

	hash[idx].release();
	compact[idx].release();
}

__global__
void resetHeapKernel(int * heap,
					 int * heap_counter,
					 Voxel * voxels)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx > NUM_SDF_BLOCKS) return;

	heap[idx] = NUM_SDF_BLOCKS - idx - 1;

	uint block_idx = idx * BLOCK_SIZE;

	for(uint i = 0; i < BLOCK_SIZE; ++i, ++block_idx)
	{
		voxels[block_idx].release();
	}

	if(idx == 0) *heap_counter = NUM_SDF_BLOCKS - 1;
}

__global__
void resetMutexKernel(int * mutex)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= NUM_BUCKETS) return;

	mutex[idx] = EntryAvailable;
}

void Map::ResetDeviceMemory() {

	dim3 block(1024);
	dim3 grid;

	grid.x = cv::divUp((int)NUM_SDF_BLOCKS, block.x);

	resetHeapKernel<<<grid, block>>>(mMemory, mUsedMem, mVoxelBlocks);

	SafeCall(cudaGetLastError());
	grid.x = cv::divUp((int)NUM_ENTRIES, block.x);

	resetHashKernel<<<grid, block>>>(mHashEntries, mVisibleEntries);

	SafeCall(cudaGetLastError());

	grid.x = cv::divUp((int)NUM_BUCKETS, block.x);

	resetMutexKernel<<<grid, block>>>(mBucketMutex);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

//void VoxelMap::resetMap()
//{
//	dim3 block(threads);
//	dim3 grid;
//
//	grid.x = divUp(NUM_SDF_BLOCKS, block.x);
//
//	resetHeapKernel<<<grid, block>>>(mHeapMem,
//									 mHeapCounter,
//									 mVoxelBlocks);
//
//	cudaSafeCall(cudaGetLastError());
//
//	grid.x = divUp(NUM_ENTRIES, block.x);
//
//	resetHashKernel<<<grid, block>>>(mHashEntries,
//									 mVisibleEntries);
//
//	cudaSafeCall(cudaGetLastError());
//
//	grid.x = divUp(NUM_BUCKETS, block.x);
//
//	resetMutexKernel<<<grid, block>>>(mBucketMutex);
//
//	cudaSafeCall(cudaGetLastError());
//	cudaSafeCall(cudaDeviceSynchronize());
//}
//
//void VoxelMap::UpdateGlobalDesc(VoxelMapDesc & desc) {
//	cudaSafeCall(cudaMemcpyToSymbol(pDesc, &desc, sizeof(VoxelMapDesc)));
//}

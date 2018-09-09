#include "rendering.h"
#include "prefixsum.h"
#include "timer.h"

struct Fusion {

	DeviceMap map;
	float invfx, invfy;
	float fx, fy, cx, cy;
	float minDepth, maxDepth;
	int cols, rows;
	Matrix3f Rview;
	Matrix3f RviewInv;
	float3 tview;

	uint* noVisibleBlocks;
	PtrStep<float> depth;
	PtrStep<uchar3> rgb;

	__device__ inline float2 project(float3& pt3d) {
		float2 pt2d;
		pt2d.x = fx * pt3d.x / pt3d.z + cx;
		pt2d.y = fy * pt3d.y / pt3d.z + cy;
		return pt2d;
	}

	__device__ inline float3 unproject(int& x, int& y, float& z) {
		float3 pt3d;
		pt3d.z = z;
		pt3d.x = z * (x - cx) * invfx;
		pt3d.y = z * (y - cy) * invfy;
		return Rview * pt3d + tview;
	}

	__device__ inline bool CheckVertexVisibility(float3 pt3d) {
		pt3d = RviewInv * (pt3d - tview);
		if (pt3d.z < 1e-3f)
			return false;
		float2 pt2d = project(pt3d);

		return pt2d.x >= 0 && pt2d.y >= 0 &&
			   pt2d.x < cols && pt2d.y < rows &&
			   pt3d.z >= minDepth && pt3d.z <= maxDepth;
	}

	__device__ inline bool CheckBlockVisibility(const int3& pos) {

		float scale = DeviceMap::blockWidth;
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

	__device__ inline void CreateBlocks() {

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= cols && y >= rows)
			return;

		float z = depth.ptr(y)[x];
		if (isnan(z) || z < DeviceMap::DepthMin ||
			z > DeviceMap::DepthMax)
			return;

		float thresh = DeviceMap::TruncateDist / 2;
		float z_near = min(DeviceMap::DepthMax, z - thresh);
		float z_far = min(DeviceMap::DepthMax, z + thresh);
		if (z_near >= z_far)
			return;

		float3 pt_near = unproject(x, y, z_near) * DeviceMap::voxelSizeInv;
		float3 pt_far = unproject(x, y, z_far) * DeviceMap::voxelSizeInv;
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

	__device__ inline void CheckFullVisibility() {

		__shared__ bool bScan;
		if (threadIdx.x == 0)
			bScan = false;
		__syncthreads();
		uint val = 0;
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.hashEntries.size) {
			HashEntry& e = map.hashEntries[x];
			if (e.ptr != EntryAvailable) {
				if (CheckBlockVisibility(e.pos)) {
					bScan = true;
					val = 1;
				}
			}
		}

		__syncthreads();
		if (bScan) {
			int offset = ComputeOffset<1024>(val, noVisibleBlocks);
			if (offset != -1 && offset < map.visibleEntries.size
					&& x < map.hashEntries.size)
				map.visibleEntries[offset] = map.hashEntries[x];
		}
	}

	__device__ inline void integrateColor() {

		if(blockIdx.x >= map.visibleEntries.size ||
		   blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry& entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;

		int locId = threadIdx.x;
		int3 block_pos = map.blockPosToVoxelPos(entry.pos);
		int3 voxel_pos = block_pos + map.localIdxToLocalPos(locId);
		float3 pos = map.voxelPosToWorldPos(voxel_pos);
		pos = RviewInv * (pos - tview);
		int2 uv = make_int2(project(pos));
		if (uv.x < 0 || uv.y < 0 || uv.x >= cols || uv.y >= rows)
			return;

		float dp = depth.ptr(uv.y)[uv.x];
		if (isnan(dp) || dp > maxDepth || dp < minDepth)
			return;

		float thresh = DeviceMap::TruncateDist;
		float sdf = dp - pos.z;
		if (sdf >= -thresh) {
			sdf = fmin(1.0f, sdf / thresh);
			float3 color = make_float3(rgb.ptr(uv.y)[uv.x]);
			Voxel & prev = map.voxelBlocks[entry.ptr + locId];
			prev.SetSdf((prev.GetSdf() * prev.sdfW + sdf) / (prev.sdfW + 1));
			prev.sdfW += 1;
			float3 color_prev = make_float3(prev.rgb);
			float3 res =  0.2f * color + 0.8f * color_prev;
			prev.rgb = make_uchar3(res);
		}
	}
};

__global__ void createBlocksKernel(Fusion fuse) {
	fuse.CreateBlocks();
}

__global__ void fuseColorKernal(Fusion fuse) {
	fuse.integrateColor();
}

__global__ void checkVisibleBlockKernel(Fusion fuse) {
	fuse.CheckFullVisibility();
}

void integrateColor(const DeviceArray2D<float> & depth,
					const DeviceArray2D<uchar3> & color,
					DeviceArray<uint> & noVisibleBlocks,
					Matrix3f Rview,
					Matrix3f RviewInv,
					float3 tview,
					DeviceMap map,
					float fx,
					float fy,
					float cx,
					float cy,
					float depthMax,
					float depthMin,
					uint * host_data) {

	int cols = depth.cols();
	int rows = depth.rows();
	noVisibleBlocks.zero();

	Fusion fuse;
	fuse.map = map;
	fuse.Rview = Rview;
	fuse.RviewInv = RviewInv;
	fuse.tview = tview;
	fuse.fx = fx;
	fuse.fy = fy;
	fuse.cx = cx;
	fuse.cy = cy;
	fuse.invfx = 1.0 / fx;
	fuse.invfy = 1.0 / fy;
	fuse.depth = depth;
	fuse.rgb = color;
	fuse.rows = rows;
	fuse.cols = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = DeviceMap::DepthMax;
	fuse.minDepth = DeviceMap::DepthMin;

	dim3 thread(32, 8);
	dim3 block(cv::divUp(cols, thread.x), cv::divUp(rows, thread.y));
	Timer::Start("debug", "createBlocksKernel");
	createBlocksKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	Timer::Stop("debug", "createBlocksKernel");

	thread = dim3(1024);
	block = dim3(cv::divUp((int) DeviceMap::NumEntries, thread.x));
	Timer::Start("debug", "checkVisibleBlockKernel");
	checkVisibleBlockKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	Timer::Stop("debug", "checkVisibleBlockKernel");

	host_data[0] = 0;
	noVisibleBlocks.download((void*) host_data);
	if (host_data[0] == 0)
		return;

	thread = dim3(512);
	block = dim3(host_data[0]);
	Timer::Start("debug", "fuseColorKernal");
	fuseColorKernal<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	Timer::Stop("debug", "fuseColorKernal");

}

__global__ void resetHashKernel(DeviceMap map) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < map.hashEntries.size) {
		map.hashEntries[x].release();
		map.visibleEntries[x].release();
	}

	if (x < DeviceMap::NumBuckets) {
		map.bucketMutex[x] = EntryAvailable;
	}
}

__global__ void resetSdfBlockKernel(DeviceMap map) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < DeviceMap::NumSdfBlocks) {
		map.heapMem[x] = DeviceMap::NumSdfBlocks - x - 1;
	}

	int blockIdx = x * DeviceMap::BlockSize3;
	for(int i = 0; i < DeviceMap::BlockSize3; ++i, ++blockIdx) {
		map.voxelBlocks[blockIdx].release();
	}

	if(x == 0) {
		map.heapCounter[0] = DeviceMap::NumSdfBlocks - 1;
		map.entryPtr[0] = 1;
	}
}

void resetDeviceMap(DeviceMap map) {

	dim3 thread(1024);
	dim3 block(cv::divUp((int) DeviceMap::NumEntries, thread.x));

	resetHashKernel<<<block, thread>>>(map);

	block.x = cv::divUp((int) DeviceMap::NumSdfBlocks, thread.x);
	resetSdfBlockKernel<<<block, thread>>>(map);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void resetKeyMapKernel(KeyMap map) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < KeyMap::maxEntries) {
		map.Keys[x].valid = false;
		map.Keys[x].obs = 0;
	}

	if(x < KeyMap::nBuckets) {
		map.Mutex[x] = EntryAvailable;
	}
}

void resetKeyMap(KeyMap map) {
	dim3 thread(1024);
	dim3 block(cv::divUp((int) KeyMap::maxEntries, thread.x));

	resetKeyMapKernel<<<block, thread>>>(map);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

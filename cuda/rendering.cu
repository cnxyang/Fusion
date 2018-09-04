#include "rendering.h"
#include "prefixsum.h"

#define minMaxSubSampling 8
#define renderingBlockSizeX 16
#define renderingBlockSizeY 16

struct Projection {

	int cols, rows;

	Matrix3f RcurrInv;
	float3 tcurr;
	float depthMax, depthMin;
	float fx, fy, cx, cy;

	uint * noRenderingBlocks;
	uint noVisibleBlocks;

	PtrSz<HashEntry> visibleBlocks;
	mutable PtrStep<float> zRangeX;
	mutable PtrStep<float> zRangeY;
	mutable PtrSz<RenderingBlock> renderingBlockList;

	__device__ inline float2 project(const float3 & pt3d) const {

		float2 pt2d;
		pt2d.x = fx * pt3d.x / pt3d.z + cx;
		pt2d.y = fy * pt3d.y / pt3d.z + cy;
		return pt2d;
	}

	__device__ inline void checkProjection(float2 & upperLeft,
			float2 & lowerRight, float2 & zRange, float2 & pt2d,
			float & z) const {

		if (z < 1e-1)
			return;
		if (upperLeft.x > pt2d.x)
			upperLeft.x = pt2d.x;
		if (lowerRight.x < pt2d.x)
			lowerRight.x = pt2d.x;
		if (upperLeft.y > pt2d.y)
			upperLeft.y = pt2d.y;
		if (lowerRight.y < pt2d.y)
			lowerRight.y = pt2d.y;
		if (zRange.x > z)
			zRange.x = z;
		if (zRange.y < z)
			zRange.y = z;
	}

	__device__ inline bool projectBlock(const int3 & pos,
			RenderingBlock & block) const {

		float2 upperLeft = make_float2(cols, rows) / minMaxSubSampling;
		float2 lowerRight = make_float2(-1, -1);
		float2 zRange = make_float2(depthMax, depthMin);
		float3 blockPos = pos * DeviceMap::blockWidth;

	#pragma unroll
		for (int corner = 0; corner < 8; ++corner) {
			float3 pt3d = blockPos;
			pt3d.x += (corner & 1) ? DeviceMap::blockWidth : 0;
			pt3d.y += (corner & 2) ? DeviceMap::blockWidth : 0;
			pt3d.z += (corner & 4) ? DeviceMap::blockWidth : 0;
			pt3d = RcurrInv * (pt3d - tcurr);
			float2 pt2d = project(pt3d) / minMaxSubSampling;
			checkProjection(upperLeft, lowerRight, zRange, pt2d, pt3d.z);
		}

		if (zRange.x < DeviceMap::DepthMin)
			zRange.x = DeviceMap::DepthMin;
		if (zRange.y < DeviceMap::DepthMin)
			return false;

		block.lowerRight.x = (int) lowerRight.x;
		block.lowerRight.y = (int) lowerRight.y;
		block.upperLeft.x = (int) upperLeft.x;
		block.upperLeft.y = (int) upperLeft.y;
		block.zRange = zRange;

		if (block.upperLeft.x < 0)
			block.upperLeft.x = 0;
		if (block.upperLeft.y < 0)
			block.upperLeft.y = 0;
		if (block.lowerRight.x >= cols)
			block.lowerRight.x = cols - 1;
		if (block.lowerRight.y >= rows)
			block.lowerRight.y = rows - 1;
		if (block.upperLeft.x > block.lowerRight.x)
			return false;
		if (block.upperLeft.y > block.lowerRight.y)
			return false;

		return true;
	}

	__device__ inline void createRenderingBlockList(int & offset,
			const RenderingBlock & block, int & nx, int & ny) const {

		for (int x = 0; x < nx; ++x)
			for (int y = 0; y < ny; ++y) {
				if (offset >= DeviceMap::MaxRenderingBlocks)
					return;

				RenderingBlock & b(renderingBlockList[offset++]);
				b.upperLeft.x = block.upperLeft.x + x * renderingBlockSizeX;
				b.upperLeft.y = block.upperLeft.y + y * renderingBlockSizeY;
				b.lowerRight.x = b.upperLeft.x + renderingBlockSizeX;
				b.lowerRight.y = b.upperLeft.y + renderingBlockSizeY;
				if (b.lowerRight.x > block.lowerRight.x)
					b.lowerRight.x = block.lowerRight.x;
				if (b.lowerRight.y > block.lowerRight.y)
					b.lowerRight.y = block.lowerRight.y;
				b.zRange = block.zRange;
			}
	}

	__device__ inline void operator()() const {

		int x = blockDim.x * blockIdx.x + threadIdx.x;
		const HashEntry & e = visibleBlocks[x];
		bool valid = false;
		uint requiredNoBlocks = 0;
		RenderingBlock block;
		int nx, ny;

		if (e.ptr != EntryAvailable && x < noVisibleBlocks) {
			valid = projectBlock(e.pos, block);
			float dx = (float) block.lowerRight.x - block.upperLeft.x + 1;
			float dy = (float) block.lowerRight.y - block.upperLeft.y + 1;
			nx = __float2int_ru(dx / renderingBlockSizeX);
			ny = __float2int_ru(dy / renderingBlockSizeY);
			if (valid) {
				requiredNoBlocks = nx * ny;
				uint totalNoBlocks = *noRenderingBlocks + requiredNoBlocks;
				if (totalNoBlocks > DeviceMap::MaxRenderingBlocks) {
					requiredNoBlocks = 0;
					valid = false;
				}
			}
		}

		int offset = ComputeOffset<1024>(requiredNoBlocks, noRenderingBlocks);
		if (valid && offset != -1)
			createRenderingBlockList(offset, block, nx, ny);
	}

	__device__ inline void fillBlocks() const {

		int x = threadIdx.x;
		int y = threadIdx.y;

		int block = blockIdx.x * 4 + blockIdx.y;
		if (block >= *noRenderingBlocks)
			return;

		RenderingBlock & b(renderingBlockList[block]);

		int xpos = b.upperLeft.x + x;
		if (xpos > b.lowerRight.x || xpos >= cols)
			return;

		int ypos = b.upperLeft.y + y;
		if (ypos > b.lowerRight.y || ypos >= rows)
			return;

		float * minPtr = & zRangeX.ptr(ypos)[xpos];
		float * maxPtr = & zRangeY.ptr(ypos)[xpos];
		printf("%f\n", b.zRange.x);
		atomicMin(minPtr, b.zRange.x);
		atomicMax(maxPtr, b.zRange.y);

		return;
	}
};

__global__ void projectBlockKernel(const Projection proj) {

	proj();
}

__global__ void fillBlocksKernel(const Projection proj) {

	proj.fillBlocks();
}

__global__ void fillDepthRangeKernel(PtrStepSz<float> zX, PtrStepSz<float> zY) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= zX.cols || y >= zX.rows)
		return;

	zX.ptr(y)[x] = 3.5;
	zY.ptr(y)[x] = 0.1;
}

bool createRenderingBlock(const DeviceArray<HashEntry> & visibleBlocks,
						  const DeviceArray2D<float> & zRangeX,
						  const DeviceArray2D<float> & zRangeY,
						  const float & depthMax,
						  const float & depthMin,
						  DeviceArray<RenderingBlock> & renderingBlockList,
						  DeviceArray<uint> & noRenderingBlocks,
						  Matrix3f RviewInv,
						  float3 tview,
						  uint noVisibleBlocks,
						  float fx,
						  float fy,
						  float cx,
						  float cy) {

	int cols = zRangeX.cols();
	int rows = zRangeX.rows();
	noRenderingBlocks.zero();
	Projection proj;
	proj.visibleBlocks = visibleBlocks;
	proj.cols = cols;
	proj.rows = rows;
	proj.RcurrInv = RviewInv;
	proj.tcurr = tview;
	proj.zRangeX = zRangeX;
	proj.zRangeY = zRangeY;
	proj.depthMax = depthMax;
	proj.depthMin = depthMin;
	proj.noRenderingBlocks = noRenderingBlocks;
	proj.noVisibleBlocks = noVisibleBlocks;
	proj.renderingBlockList = renderingBlockList;

	dim3 block, thread;
	thread.x = 32; block.y = 8;
	block.x = cv::divUp(cols, thread.x);
	block.y = cv::divUp(rows, thread.y);

	fillDepthRangeKernel<<<block, thread>>>(zRangeX, zRangeY);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	thread = dim3(1024);
	block = dim3(cv::divUp((int) noVisibleBlocks, block.x));

	projectBlockKernel<<<block, thread>>>(proj);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	uint totalBlocks;
	noRenderingBlocks.download((void*) &totalBlocks);

	if (totalBlocks == 0) {
		return false;
	}

	thread.x = thread.y = 16;
	block.x = (uint)ceil((float)totalBlocks / 4);
	block.y = 4;

	fillBlocksKernel<<<block, thread>>>(proj);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	return true;
}

struct Rendering {

	int cols, rows;
	DeviceMap map;
	mutable PtrStep<float4> vmap;
	mutable PtrStep<float3> nmap;
	PtrStep<float> zRangeX;
	PtrStep<float> zRangeY;
	float invfx, invfy, cx, cy;
	Matrix3f Rview, RviewInv;
	float3 tview;

	__device__ inline float readSdf(float3 & pt3d) {
		Voxel voxel = map.FindVoxel(pt3d);
		if (voxel.sdfW == 0)
			return 1.f;
		return voxel.GetSdf();
	}

	__device__ inline float readSdfInterped(float3 & pt3d) {
		float res1, res2, v1, v2;
		float3 coeff;
		coeff = pt3d - floor(pt3d);
		int3 vpos = make_int3(pt3d + 0.5);

		v1 = map.FindVoxel(pt3d + make_float3(0, 0, 0)).GetSdf();
		v2 = map.FindVoxel(pt3d + make_float3(1, 0, 0)).GetSdf();
		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = map.FindVoxel(pt3d + make_float3(0, 1, 0)).GetSdf();
		v2 = map.FindVoxel(pt3d + make_float3(1, 1, 0)).GetSdf();
		res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		v1 = map.FindVoxel(pt3d + make_float3(0, 0, 1)).GetSdf();
		v2 = map.FindVoxel(pt3d + make_float3(1, 0, 1)).GetSdf();
		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = map.FindVoxel(pt3d + make_float3(0, 1, 1)).GetSdf();
		v2 = map.FindVoxel(pt3d + make_float3(1, 1, 1)).GetSdf();
		res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		return (1.0f - coeff.z) * res1 + coeff.z * res2;
	}

	__device__ inline void operator()() {

		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x >= cols || y >= rows)
			return;

		int2 locId;
		locId.x = __float2int_rd((float) x / minMaxSubSampling);
		locId.y = __float2int_rd((float) y / minMaxSubSampling);

		float2 zRange;
		zRange.x = zRangeX.ptr(locId.y)[locId.x];
		zRange.y = zRangeY.ptr(locId.y)[locId.x];
		if(zRange.y < 1e-3 || zRange.x < 1e-3 || isnan(zRange.x))
			return;

		float sdf = 1.0f;
		float stepScale = DeviceMap::TruncateDist * DeviceMap::voxelSizeInv;

		float3 pt3d;
		pt3d.z = zRange.x;
		pt3d.x = pt3d.z * (x - cx) * invfx;
		pt3d.y = pt3d.z * (y - cy) * invfy;
		float dist_s = norm(pt3d) * DeviceMap::voxelSizeInv;
		float3 block_s = Rview * pt3d + tview;
		block_s = block_s * DeviceMap::voxelSizeInv;

		pt3d.z = zRange.y;
		pt3d.x = pt3d.z * (x - cx) * invfx;
		pt3d.y = pt3d.z * (y - cy) * invfy;
		float dist_e = norm(pt3d) * DeviceMap::voxelSizeInv;
		float3 block_e = Rview * pt3d + tview;
		block_e = block_e * DeviceMap::voxelSizeInv;

		float3 dir = normalised(block_e - block_s);
		float3 result = block_s;

		bool found_pt = false;
		float step;
		while (dist_s < dist_e) {
			int3 blockPos = map.voxelPosToBlockPos(make_int3(result));
			HashEntry b = map.FindEntry(blockPos);
			if(b.ptr != EntryAvailable) {
				sdf = readSdf(result);
				if(sdf <= 0.1f && sdf >= -0.5f) {
					sdf = readSdfInterped(result);
				}

				if(sdf <= 0.0f)
					break;

				step = max(sdf * stepScale, 1.0f);
			}
			else
				step = DeviceMap::BlockSize;

			result += step * dir;
			dist_s += step;
		}

		if(sdf <= 0.0f) {
			step = sdf * stepScale;
			result += step * dir;

			sdf = readSdfInterped(result);

			step = sdf * stepScale;
			result += step * dir;
			found_pt = true;
		}

		if(found_pt) {
			result = RviewInv * (result - tview);
			vmap.ptr(y)[x] = make_float4(result, 1.0);
		}
		else {
			vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		}
	}
};

__global__ void RayCastKernel(Rendering cast) {
	cast();
}

void RayCast(DeviceMap map,
			 DeviceArray2D<float4> & vmap,
			 DeviceArray2D<float3> & nmap,
			 DeviceArray2D<float> & zRangeX,
			 DeviceArray2D<float> & zRangeY,
			 Matrix3f Rview,
			 Matrix3f RviewInv,
			 float3 tview,
			 float invfx,
			 float invfy,
			 float cx,
			 float cy) {

	int cols = vmap.cols();
	int rows = vmap.rows();

	Rendering cast;
	cast.cols = cols;
	cast.rows = rows;

	cast.map = map;
	cast.vmap = vmap;
	cast.nmap = nmap;
	cast.zRangeX = zRangeX;
	cast.zRangeY = zRangeY;
	cast.invfx = invfx;
	cast.invfy = invfy;
	cast.cx = cx;
	cast.cy = cy;
	cast.Rview = Rview;
	cast.RviewInv = RviewInv;
	cast.tview = tview;

	dim3 block, thread;
	thread.x = 32;
	thread.y = 8;
	block.x = cv::divUp(cols, thread.x);
	block.y = cv::divUp(rows, thread.y);

	RayCastKernel<<<block, thread>>>(cast);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

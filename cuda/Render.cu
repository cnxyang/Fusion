#include "Map.hpp"
#include "DeviceStruct.h"

struct HashRayCaster
{
	DeviceMap map;
	uint num_blocks;

	mutable PtrStep<float4> vmap;
	mutable PtrStep<float3> nmap;
	mutable PtrStep<uchar4> rendering;

	mutable PtrStep<float> minDepthMap;
	mutable PtrStep<float> maxDepthMap;

	RenderingBlock * renderingBlockList;
	uint * noTotalBlocks;

	static const int minmaximg_subsample = 8;
	static const int renderingBlockSizeX = 16;
	static const int renderingBlockSizeY = 16;

	float fx, fy ,cx, cy;
    uint width;
	uint height;
	Matrix3f R_curr;
	Matrix3f R_inv;
	float3 t_curr;

	__device__ __forceinline__
	bool ProjectSingleBlock(const int3 & blockPos,
						    int2 & upperLeft,
						    int2 & lowerRight,
						    float2 & zRange)
	{
		upperLeft = make_int2(width, height) / minmaximg_subsample;
		lowerRight = make_int2(-1, -1);
		zRange = make_float2(DEPTH_MAX, DEPTH_MIN);
		for (int corner = 0; corner < 8; ++corner)
		{
			// project all 8 corners down to 2D image
			int3 tmp = blockPos;
			tmp.x += (corner & 1) ? 1 : 0;
			tmp.y += (corner & 2) ? 1 : 0;
			tmp.z += (corner & 4) ? 1 : 0;
			float3 pt3d = tmp * BLOCK_DIM * VOXEL_SIZE;
			pt3d = R_inv * (pt3d - t_curr);
			if (pt3d.z < 1e-6) continue;

			float2 pt2d;
			pt2d.x = (fx * pt3d.x / pt3d.z + cx) / minmaximg_subsample;
			pt2d.y = (fy * pt3d.y / pt3d.z + cy) / minmaximg_subsample;

			// remember bounding box, zmin and zmax
			if (upperLeft.x > floor(pt2d.x)) upperLeft.x = (int)floor(pt2d.x);
			if (lowerRight.x < ceil(pt2d.x)) lowerRight.x = (int)ceil(pt2d.x);
			if (upperLeft.y > floor(pt2d.y)) upperLeft.y = (int)floor(pt2d.y);
			if (lowerRight.y < ceil(pt2d.y)) lowerRight.y = (int)ceil(pt2d.y);
			if (zRange.x > pt3d.z) zRange.x = pt3d.z;
			if (zRange.y < pt3d.z) zRange.y = pt3d.z;
		}

		// do some sanity checks and respect image bounds
		if (upperLeft.x < 0) upperLeft.x = 0;
		if (upperLeft.y < 0) upperLeft.y = 0;
		if (lowerRight.x >= width) lowerRight.x = width - 1;
		if (lowerRight.y >= height) lowerRight.y = height - 1;
		if (upperLeft.x > lowerRight.x) return false;
		if (upperLeft.y > lowerRight.y) return false;
		//if (zRange.y <= VERY_CLOSE) return false; never seems to happen
		if (zRange.x < DEPTH_MIN) zRange.x = DEPTH_MIN;
		if (zRange.y < DEPTH_MIN) return false;

		return true;
	}

	template <typename T> __device__
	int computePrefixSum_device(uint element, T *sum, int localSize, int localId)
	{
		// TODO: should be localSize...
		__shared__ uint prefixBuffer[16 * 16];
		__shared__ uint groupOffset;

		if(localId == 0)
			memset(prefixBuffer, 0, sizeof(uint) * 16 * 16);
		__syncthreads();

		prefixBuffer[localId] = element;
		__syncthreads();

		int s1, s2;

		for (s1 = 1, s2 = 1; s1 < localSize; s1 <<= 1)
		{
			s2 |= s1;
			if ((localId & s2) == s2) prefixBuffer[localId] += prefixBuffer[localId - s1];
			__syncthreads();
		}

		for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
		{
			if (localId != localSize - 1 && (localId & s2) == s2) prefixBuffer[localId + s1] += prefixBuffer[localId];
			__syncthreads();
		}

		if (localId == 0 && prefixBuffer[localSize - 1] > 0) groupOffset = atomicAdd(sum, prefixBuffer[localSize - 1]);
		__syncthreads();

		int offset;// = groupOffset + prefixBuffer[localId] - 1;
		if (localId == 0) {
			if (prefixBuffer[localId] == 0) offset = -1;
			else offset = groupOffset;
		} else {
			if (prefixBuffer[localId] == prefixBuffer[localId - 1]) offset = -1;
			else offset = groupOffset + prefixBuffer[localId-1];
		}

		return offset;
	}

	__device__ inline
	void CreateRenderingBlocks(int offset,
							   const int2 & upperLeft,
							   int2 & lowerRight,
							   const float2 & zRange)
	{
		// split bounding box into 16x16 pixel rendering blocks
		for (int by = 0; by < ceil((float)(1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by) {
			for (int bx = 0; bx < ceil((float)(1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx) {
				if (offset >= MAX_RENDERING_BLOCKS) return;
				//for each rendering block: add it to the list
				RenderingBlock & b(renderingBlockList[offset++]);
				b.upperLeft.x = upperLeft.x + bx * renderingBlockSizeX;
				b.upperLeft.y = upperLeft.y + by * renderingBlockSizeY;
				b.lowerRight.x = upperLeft.x + (bx + 1) * renderingBlockSizeX - 1;
				b.lowerRight.y = upperLeft.y + (by + 1) * renderingBlockSizeY - 1;
				if (b.lowerRight.x > lowerRight.x) b.lowerRight.x = lowerRight.x;
				if (b.lowerRight.y > lowerRight.y) b.lowerRight.y = lowerRight.y;
				b.zRange = zRange;
			}
		}
	}

	__device__
	void projectAndSplitBlocks_device()
	{
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
		int out_offset = computePrefixSum_device<uint>(requiredNumBlocks, noTotalBlocks, blockDim.x, threadIdx.x);
		if (!validProjection) return;
		if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

		CreateRenderingBlocks(out_offset, upperLeft, lowerRight, zRange);
	}

	__device__
	void fillBlocks_device()
	{
		int x = threadIdx.x;
		int y = threadIdx.y;
		int block = blockIdx.x * 4 + blockIdx.y;
		if (block >= *noTotalBlocks) return;

		const RenderingBlock & b(renderingBlockList[block]);
		int xpos = b.upperLeft.x + x;
		if (xpos > b.lowerRight.x) return;
		int ypos = b.upperLeft.y + y;
		if (ypos > b.lowerRight.y) return;

		float * minData = &minDepthMap.ptr(ypos)[xpos];
		float * maxData = &maxDepthMap.ptr(ypos)[xpos];

		atomicMin(minData, b.zRange.x);
		atomicMax(maxData, b.zRange.y);
		return;
	}

	__device__ inline
	bool castRay(float4 & pt_out, int x, int y, float mu, float oneOverVoxelSize, float mind, float maxd)
	{
		float3 pt_camera_f, pt_block_s, pt_block_e, rayDirection, pt_result;
		bool pt_found;

		float sdfValue = 1.0f, confidence;
		float totalLength, stepLength, totalLengthMax, stepScale;

		stepScale = TRUNC_DIST * oneOverVoxelSize;

		pt_camera_f.z = mind;
		pt_camera_f.x = pt_camera_f.z * ((float(x) - cx) / fx);
		pt_camera_f.y = pt_camera_f.z * ((float(y) - cy) / fy);
		totalLength = norm(pt_camera_f) * oneOverVoxelSize;
		pt_block_s = (R_curr * pt_camera_f + t_curr) * oneOverVoxelSize;

		pt_camera_f.z = maxd;
		pt_camera_f.x = pt_camera_f.z * ((float(x) - cx) / fx);
		pt_camera_f.y = pt_camera_f.z * ((float(y) - cy) / fy);
		totalLengthMax = norm(pt_camera_f) * oneOverVoxelSize;
		pt_block_e = (R_curr * pt_camera_f + t_curr) * oneOverVoxelSize;

		rayDirection = normalised(pt_block_e - pt_block_s);

		pt_result = pt_block_s;

		while (totalLength < totalLengthMax)
		{
			HashEntry block = map.searchHashEntry(map.voxelPosToBlockPos(make_int3(pt_result)));

			if(block.ptr == EntryAvailable)
			{
				stepLength = BLOCK_DIM;
			}
			else
			{
				sdfValue = readFromSDF_float_uninterpolated(pt_result);

				if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f))
				{
						sdfValue = readFromSDF_float_interpolated(pt_result);
				}


				if (sdfValue <= 0.0f)
					break;

				stepLength = max(sdfValue * stepScale, 1.0f);
			}

			pt_result += stepLength * rayDirection; totalLength += stepLength;
		}

		if (sdfValue <= 0.0f)
		{
			stepLength = sdfValue * stepScale;
			pt_result += stepLength * rayDirection;

			sdfValue = readWithConfidenceFromSDF_float_interpolated(confidence, pt_result);

			stepLength = sdfValue * stepScale;
			pt_result += stepLength * rayDirection;

			pt_found = true;
		}
		else
			pt_found = false;

		pt_out = make_float4(R_inv * (pt_result / oneOverVoxelSize - t_curr), 1.f);
		if (pt_found)
			pt_out.w = confidence + 1.0f;
		else
			pt_out.w = 0.0f;

		return pt_found;
	}

	__device__ inline
	float readFromSDF_float_uninterpolated(float3 point)
	{
		Voxel voxel =  map.searchVoxel(point);
		if(voxel.sdfW == 0) return 1.f;
		return voxel.sdf;
	}

	__device__ inline
	float readWithConfidenceFromSDF_float_interpolated(float &confidence, float3 point)
	{
		float res1, res2, v1, v2;
		float res1_c, res2_c, v1_c, v2_c;

		float3 coeff; Voxel voxel;
		coeff = point - floor(point);

		voxel = map.searchVoxel(point + make_float3(0, 0, 0)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = map.searchVoxel(point + make_float3(1, 0, 0)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
		res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

		voxel = map.searchVoxel(point + make_float3(0, 1, 0)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = map.searchVoxel(point + make_float3(1, 1, 0)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
		res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

		voxel = map.searchVoxel(point + make_float3(0, 0, 1)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = map.searchVoxel(point + make_float3(1, 0, 1)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
		res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

		voxel = map.searchVoxel(point + make_float3(0, 1, 1)); v1 = voxel.sdf; v1_c = voxel.sdfW;
		voxel = map.searchVoxel(point + make_float3(1, 1, 1)); v2 = voxel.sdf; v2_c = voxel.sdfW;
		res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
		res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

		confidence = (1.0f - coeff.z) * res1_c + coeff.z * res2_c;

		return (1.0f - coeff.z) * res1 + coeff.z * res2;
	}

	__device__
	float readFromSDF_float_interpolated(const float3 & pos)
	{
		float res1, res2, v1, v2;
		float3 coeff;
		coeff = pos - floor(pos);
		int3 vpos = make_int3(pos + 0.5);

		v1 = map.searchVoxel(pos + make_float3(0, 0, 0)).sdf;
		v2 = map.searchVoxel(pos + make_float3(1, 0, 0)).sdf;
		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = map.searchVoxel(pos + make_float3(0, 1, 0)).sdf;
		v2 = map.searchVoxel(pos + make_float3(1, 1, 0)).sdf;
		res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		v1 = map.searchVoxel(pos + make_float3(0, 0, 1)).sdf;
		v2 = map.searchVoxel(pos + make_float3(1, 0, 1)).sdf;
		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

		v1 = map.searchVoxel(pos + make_float3(0, 1, 1)).sdf;
		v2 = map.searchVoxel(pos + make_float3(1, 1, 1)).sdf;
		res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

		return (1.0f - coeff.z) * res1 + coeff.z * res2;
	}


	template<bool flipNormals>
	__device__ void ComputeNormalMap()
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);
		if (x >= width || y >= height) return;

		float3 outNormal;
		float angle;
		float4 point = vmap.ptr(y)[x];
		bool foundPoint = point.w > 0.0f;

		computeNormalAndAngle<true, flipNormals>(foundPoint, x, y, outNormal, angle);

		if (foundPoint)
		{
			nmap.ptr(y)[x] = outNormal;
		}
	}

	template <bool useSmoothing, bool flipNormals>
	__device__ inline void computeNormalAndAngle(bool & foundPoint, int &x, int &y, float3 & outNormal, float& angle)
	{
		if (!foundPoint) return;

		float4 xp1_y, xm1_y, x_yp1, x_ym1;

		if (useSmoothing)
		{
			if (y <= 2 || y >= height - 3 || x <= 2 || x >= width - 3) { foundPoint = false; return; }

			xp1_y = vmap.ptr(y)[x + 2], x_yp1 = vmap.ptr(y + 2)[x];
			xm1_y = vmap.ptr(y)[x - 2], x_ym1 = vmap.ptr(y - 2)[x];
		}
		else
		{
			if (y <= 1 || y >= height - 2 || x <= 1 || x >= width - 2) { foundPoint = false; return; }

			xp1_y = vmap.ptr(y)[x + 1], x_yp1 = vmap.ptr(y + 1)[x];
			xm1_y = vmap.ptr(y)[x - 1], x_ym1 = vmap.ptr(y - 1)[x];
		}

		float4 diff_x = make_float4(0.0f, 0.0f, 0.0f, 0.0f), diff_y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		bool doPlus1 = false;
		if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0) doPlus1 = true;
		else
		{
			diff_x = xp1_y - xm1_y, diff_y = x_yp1 - x_ym1;

			float length_diff = fmax(diff_x.x * diff_x.x + diff_x.y * diff_x.y + diff_x.z * diff_x.z,
				diff_y.x * diff_y.x + diff_y.y * diff_y.y + diff_y.z * diff_y.z);

			if (length_diff * VOXEL_SIZE * VOXEL_SIZE > (0.15f * 0.15f)) doPlus1 = true;
		}

		if (doPlus1)
		{
			if (useSmoothing)
			{
				xp1_y = vmap.ptr(y)[x + 1]; x_yp1 = vmap.ptr(y + 1)[x];
				xm1_y = vmap.ptr(y)[x - 1]; x_ym1 = vmap.ptr(y - 2)[x];
				diff_x = xp1_y - xm1_y; diff_y = x_yp1 - x_ym1;
			}

			if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)
			{
				foundPoint = false;
				return;
			}
		}

		outNormal.x = -(diff_x.y * diff_y.z - diff_x.z*diff_y.y);
		outNormal.y = -(diff_x.z * diff_y.x - diff_x.x*diff_y.z);
		outNormal.z = -(diff_x.x * diff_y.y - diff_x.y*diff_y.x);

		if (flipNormals) outNormal = -outNormal;

		float normScale = 1.0f / sqrt(outNormal.x * outNormal.x + outNormal.y * outNormal.y + outNormal.z * outNormal.z);
		outNormal = outNormal * normScale;
	}

	__device__ inline
	void drawPixelNormal(uchar4 & dest, float3 & normal_obj)
	{
		dest.x = (unsigned char) ((0.3f + (-normal_obj.x + 1.0f) * 0.35f) * 255.0f);
		dest.y = (unsigned char) ((0.3f + (-normal_obj.y + 1.0f) * 0.35f) * 255.0f);
		dest.z = (unsigned char) ((0.3f + (-normal_obj.z + 1.0f) * 0.35f) * 255.0f);
	}


	__device__
	void operator()()
	{
		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
		const uint y = blockIdx.y * blockDim.y + threadIdx.y;


		vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		nmap.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));

		int2 locId = make_int2(__float2int_rd((float)x / minmaximg_subsample),
							   __float2int_rd((float)y / minmaximg_subsample));

		float time_min = minDepthMap.ptr(locId.y)[locId.x];
		float time_max = maxDepthMap.ptr(locId.y)[locId.x];

		if(x >= width || y >= height)
			return;

		if (time_min == 0 || time_min == __int_as_float(0x7fffffff)) return;
		if (time_max == 0 || time_max == __int_as_float(0x7fffffff)) return;

		time_min = max(time_min, DEPTH_MIN);
		time_max = min(time_max, DEPTH_MAX);

		float4 pt_out;

		if(castRay(pt_out, x, y, VOXEL_SIZE, 1 / VOXEL_SIZE, time_min, time_max))
		{
			vmap.ptr(y)[x] = pt_out;
		}
	}
};

__global__ void
projectAndSplitBlocksKernel(HashRayCaster hrc) {
	hrc.projectAndSplitBlocks_device();
}

__global__ void
fillBlocksKernel(HashRayCaster hrc) {
	hrc.fillBlocks_device();
}

__global__ void
hashRayCastKernel(HashRayCaster hrc) {
	hrc();
}

__global__ void
computeNormalAndAngleKernel(HashRayCaster hrc) {
	hrc.ComputeNormalMap<false>();
}

__global__ void
Memset_device(PtrStepSz<float> a2d, float val) {
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= a2d.cols || y >= a2d.rows)
		return;

	a2d.ptr(y)[x] = val;
}

void Map::RenderMap(Rendering& render, int num_occupied_blocks) {

	if(num_occupied_blocks == 0)
		return;

	render.VMap.create(render.cols, render.rows);
	render.NMap.create(render.cols, render.rows);
	render.Render.create(render.cols, render.rows);

	DeviceArray<RenderingBlock> RenderingBlockList(MAX_RENDERING_BLOCKS);
	DeviceArray<uint> noTotalBlocks(1);
	noTotalBlocks.zero();
	DeviceArray2D<float> DepthMapMin(render.cols, render.rows);
	DeviceArray2D<float> DepthMapMax(render.cols , render.rows);

	dim3 b(8, 8);
	dim3 g(cv::divUp(DepthMapMin.cols(), b.x), cv::divUp(DepthMapMin.rows(), b.y));
	Memset_device<<<g, b>>>(DepthMapMin, DEPTH_MAX);
	Memset_device<<<g, b>>>(DepthMapMax, DEPTH_MIN);

	HashRayCaster hrc;
	hrc.map = *this;
	hrc.num_blocks = num_occupied_blocks;
	hrc.fx = render.fx;
	hrc.fy = render.fy;
	hrc.cx = render.cx;
	hrc.cy = render.cy;
	hrc.width = render.cols;
	hrc.height = render.rows;
	hrc.R_curr = render.Rview;
	hrc.R_inv = render.invRview;
	hrc.t_curr = render.tview;
	hrc.renderingBlockList = RenderingBlockList;
	hrc.minDepthMap = DepthMapMin;
	hrc.maxDepthMap = DepthMapMax;
	hrc.map = *this;
	hrc.noTotalBlocks = noTotalBlocks;
	hrc.vmap = render.VMap;
	hrc.nmap = render.NMap;
	hrc.rendering = render.Render;

	const dim3 block(256);
	const dim3 grid(cv::divUp(num_occupied_blocks, block.x));

	projectAndSplitBlocksKernel<<<grid, block>>>(hrc);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	uint totalBlocks;
	noTotalBlocks.download((void*)&totalBlocks);
	if (totalBlocks == 0 || totalBlocks >= MAX_RENDERING_BLOCKS)
		return;

	dim3 blocks = dim3(16, 16);
	dim3 grids = dim3((unsigned int) ceil((float) totalBlocks / 4.0f), 4);

	fillBlocksKernel<<<grids, blocks>>>(hrc);

    SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	dim3 block1(32, 8);
	dim3 grid1(cv::divUp(render.cols, block1.x), cv::divUp(render.rows, block1.y));

	hashRayCastKernel<<<grid1, block1>>>(hrc);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	computeNormalAndAngleKernel<<<grid1, block1>>>(hrc);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	renderImage(render.VMap, render.NMap, make_float3(0), render.Render);
}

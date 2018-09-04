#include "Mapping.hpp"
#include "Timer.hpp"
#include "device_array.hpp"
#include "device_math.hpp"
#include "device_mapping.cuh"

#define CUDA_KERNEL __global__
#define DEV_FUNC __device__

//#define Render_DownScaleRatio   8
//#define Render_RenderBlockSize  16

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

//struct HashRayCaster {
//
//	DeviceMap map;
//	float fx, fy, cx, cy;
//	int cols, rows;
//	Matrix3f Rot, invRot;
//	float3 trans;
//
//	uint nVoxelBlocks;
//
//	mutable PtrStep<float4> vmap;
//	mutable PtrStep<float3> nmap;
//	mutable PtrStep<uchar4> rendering;
//	mutable PtrStep<float> minDepthMap;
//	mutable PtrStep<float> maxDepthMap;
//	PtrSz<RenderingBlock> renderingBlockList;
//	uint * nRenderingBlocks;
//
//	static const int minmaximg_subsample = 8;
//	static const int renderingBlockSizeX = 16;
//	static const int renderingBlockSizeY = 16;
//
//	DEV_FUNC float2 ProjectVertex(const float3& pt) {
//		float2 pt2d;
//		pt2d.x = fx * pt.x / pt.z + cx;
//		pt2d.y = fy * pt.y / pt.z + cy;
//		return pt2d;
//	}
//
//	DEV_FUNC bool ProjectBlock(const int3& blockPos, int2& upperLeft,
//			int2& lowerRight, float2& zRange) {
//
//		upperLeft = make_int2(cols, rows) / minmaximg_subsample;
//		lowerRight = make_int2(-1, -1);
//		zRange = make_float2(DeviceMap::DepthMax, DeviceMap::DepthMin);
//		for (int corner = 0; corner < 8; ++corner) {
//			int3 tmp = blockPos;
//			tmp.x += (corner & 1) ? 1 : 0;
//			tmp.y += (corner & 2) ? 1 : 0;
//			tmp.z += (corner & 4) ? 1 : 0;
//			float3 pt3d = tmp * DeviceMap::BlockSize * DeviceMap::VoxelSize;
//			pt3d = invRot * (pt3d - trans);
//			if (pt3d.z < 1e-6)
//				continue;
//
//			float2 pt2d = ProjectVertex(pt3d) / minmaximg_subsample;
//
//			if (upperLeft.x > floor(pt2d.x))
//				upperLeft.x = (int) floor(pt2d.x);
//			if (lowerRight.x < ceil(pt2d.x))
//				lowerRight.x = (int) ceil(pt2d.x);
//			if (upperLeft.y > floor(pt2d.y))
//				upperLeft.y = (int) floor(pt2d.y);
//			if (lowerRight.y < ceil(pt2d.y))
//				lowerRight.y = (int) ceil(pt2d.y);
//			if (zRange.x > pt3d.z)
//				zRange.x = pt3d.z;
//			if (zRange.y < pt3d.z)
//				zRange.y = pt3d.z;
//		}
//
//		// do some sanity checks and respect image bounds
//		if (upperLeft.x < 0)
//			upperLeft.x = 0;
//		if (upperLeft.y < 0)
//			upperLeft.y = 0;
//		if (lowerRight.x >= cols)
//			lowerRight.x = cols - 1;
//		if (lowerRight.y >= rows)
//			lowerRight.y = rows - 1;
//		if (upperLeft.x > lowerRight.x)
//			return false;
//		if (upperLeft.y > lowerRight.y)
//			return false;
//		if (zRange.x < DeviceMap::DepthMin)
//			zRange.x = DeviceMap::DepthMin;
//		if (zRange.y < DeviceMap::DepthMin)
//			return false;
//
//		return true;
//	}
//
//	__device__ inline
//	void CreateRenderingBlocks(int offset, const int2 & upperLeft,
//			int2 & lowerRight, const float2 & zRange) {
//		// split bounding box into 16x16 pixel rendering blocks
//		for (int by = 0;
//				by
//						< ceil(
//								(float) (1 + lowerRight.y - upperLeft.y)
//										/ renderingBlockSizeY); ++by) {
//			for (int bx = 0;
//					bx
//							< ceil(
//									(float) (1 + lowerRight.x - upperLeft.x)
//											/ renderingBlockSizeX); ++bx) {
//				if (offset >= DeviceMap::MaxRenderingBlocks)
//					return;
//				//for each rendering block: add it to the list
//				RenderingBlock & b(renderingBlockList[offset++]);
//				b.upperLeft.x = upperLeft.x + bx * renderingBlockSizeX;
//				b.upperLeft.y = upperLeft.y + by * renderingBlockSizeY;
//				b.lowerRight.x = upperLeft.x + (bx + 1) * renderingBlockSizeX
//						- 1;
//				b.lowerRight.y = upperLeft.y + (by + 1) * renderingBlockSizeY
//						- 1;
//				if (b.lowerRight.x > lowerRight.x)
//					b.lowerRight.x = lowerRight.x;
//				if (b.lowerRight.y > lowerRight.y)
//					b.lowerRight.y = lowerRight.y;
//				b.zRange = zRange;
//			}
//		}
//	}
//
//	__device__
//	void projectAndSplitBlocks_device() {
//		int idx = threadIdx.x + blockDim.x * blockIdx.x;
//
//		const HashEntry & entry(map.visibleEntries[idx]);
//
//		if (entry.ptr == EntryAvailable || idx >= nVoxelBlocks)
//			return;
//
//		int2 upperLeft;
//		int2 lowerRight;
//		float2 zRange;
//
//		bool validProjection = false;
//
//		validProjection = ProjectBlock(entry.pos, upperLeft, lowerRight,
//				zRange);
//
//		int2 requiredRenderingBlocks = make_int2(
//				ceilf(
//						(float) (lowerRight.x - upperLeft.x + 1)
//								/ renderingBlockSizeX),
//				ceilf(
//						(float) (lowerRight.y - upperLeft.y + 1)
//								/ renderingBlockSizeY));
//
//		uint requiredNumBlocks = 0;
//		if (validProjection) {
//			requiredNumBlocks = requiredRenderingBlocks.x
//					* requiredRenderingBlocks.y;
//			if (*nRenderingBlocks + requiredNumBlocks
//					>= DeviceMap::MaxRenderingBlocks)
//				requiredNumBlocks = 0;
//		}
//
//		int out_offset = ComputeOffset<256>(requiredNumBlocks,
//				nRenderingBlocks);
//		if (!validProjection || out_offset == -1
//				|| *nRenderingBlocks + out_offset
//						>= DeviceMap::MaxRenderingBlocks)
//			return;
//
//		CreateRenderingBlocks(out_offset, upperLeft, lowerRight, zRange);
//	}
//
//	__device__
//	void fillBlocks_device() {
//		int x = threadIdx.x;
//		int y = threadIdx.y;
//		int block = blockIdx.x * 4 + blockIdx.y;
//		if (block >= *nRenderingBlocks)
//			return;
//
//		const RenderingBlock & b(renderingBlockList[block]);
//		int xpos = b.upperLeft.x + x;
//		if (xpos > b.lowerRight.x)
//			return;
//		int ypos = b.upperLeft.y + y;
//		if (ypos > b.lowerRight.y)
//			return;
//
//		float * minData = &minDepthMap.ptr(ypos)[xpos];
//		float * maxData = &maxDepthMap.ptr(ypos)[xpos];
//
//		atomicMin(minData, b.zRange.x);
//		atomicMax(maxData, b.zRange.y);
//		return;
//	}
//
//	__device__ inline
//	bool castRay(float4 & pt_out, int x, int y, float mu,
//			float oneOverVoxelSize, float mind, float maxd) {
//		float3 pt_camera_f, pt_block_s, pt_block_e, rayDirection, pt_result;
//		bool pt_found;
//
//		float sdfValue = 1.0f, confidence;
//		float totalLength, stepLength, totalLengthMax, stepScale;
//
//		stepScale = DeviceMap::TruncateDist * oneOverVoxelSize;
//
//		pt_camera_f.z = mind;
//		pt_camera_f.x = pt_camera_f.z * ((float(x) - cx) / fx);
//		pt_camera_f.y = pt_camera_f.z * ((float(y) - cy) / fy);
//		totalLength = norm(pt_camera_f) * oneOverVoxelSize;
//		pt_block_s = (Rot * pt_camera_f + trans) * oneOverVoxelSize;
//
//		pt_camera_f.z = maxd;
//		pt_camera_f.x = pt_camera_f.z * ((float(x) - cx) / fx);
//		pt_camera_f.y = pt_camera_f.z * ((float(y) - cy) / fy);
//		totalLengthMax = norm(pt_camera_f) * oneOverVoxelSize;
//		pt_block_e = (Rot * pt_camera_f + trans) * oneOverVoxelSize;
//
//		rayDirection = normalised(pt_block_e - pt_block_s);
//
//		pt_result = pt_block_s;
//
//		while (totalLength < totalLengthMax) {
//			HashEntry block = map.FindEntry(
//					map.voxelPosToBlockPos(make_int3(pt_result)));
//
//			if (block.ptr == EntryAvailable) {
//				stepLength = DeviceMap::BlockSize;
//			} else {
//				sdfValue = readFromSDF_float_uninterpolated(pt_result);
//
//				if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
//					sdfValue = readFromSDF_float_interpolated(pt_result);
//				}
//
//				if (sdfValue <= 0.0f)
//					break;
//
//				stepLength = max(sdfValue * stepScale, 1.0f);
//			}
//
//			pt_result += stepLength * rayDirection;
//			totalLength += stepLength;
//		}
//
//		if (sdfValue <= 0.0f) {
//			stepLength = sdfValue * stepScale;
//			pt_result += stepLength * rayDirection;
//
//			sdfValue = readWithConfidenceFromSDF_float_interpolated(confidence,
//					pt_result);
//
//			stepLength = sdfValue * stepScale;
//			pt_result += stepLength * rayDirection;
//
//			pt_found = true;
//		} else
//			pt_found = false;
//
//		pt_out = make_float4(invRot * (pt_result / oneOverVoxelSize - trans),
//				1.f);
////		pt_out = make_float4(pt_result / oneOverVoxelSize, 1.f);
//		if (pt_found)
//			pt_out.w = confidence + 1.0f;
//		else
//			pt_out.w = 0.0f;
//
//		return pt_found;
//	}
//
//	__device__ inline
//	float readFromSDF_float_uninterpolated(float3 point) {
//		Voxel voxel = map.FindVoxel(point);
//		if (voxel.sdfW == 0)
//			return 1.f;
//		return voxel.GetSdf();
//	}
//
//	__device__ inline
//	float readWithConfidenceFromSDF_float_interpolated(float &confidence,
//			float3 point) {
//		float res1, res2, v1, v2;
//		float res1_c, res2_c, v1_c, v2_c;
//
//		float3 coeff;
//		Voxel voxel;
//		coeff = point - floor(point);
//
//		voxel = map.FindVoxel(point + make_float3(0, 0, 0));
//		v1 = voxel.GetSdf();
//		v1_c = voxel.sdfW;
//		voxel = map.FindVoxel(point + make_float3(1, 0, 0));
//		v2 = voxel.GetSdf();
//		v2_c = voxel.sdfW;
//		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
//		res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;
//
//		voxel = map.FindVoxel(point + make_float3(0, 1, 0));
//		v1 = voxel.GetSdf();
//		v1_c = voxel.sdfW;
//		voxel = map.FindVoxel(point + make_float3(1, 1, 0));
//		v2 = voxel.GetSdf();
//		v2_c = voxel.sdfW;
//		res1 = (1.0f - coeff.y) * res1
//				+ coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
//		res1_c = (1.0f - coeff.y) * res1_c
//				+ coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);
//
//		voxel = map.FindVoxel(point + make_float3(0, 0, 1));
//		v1 = voxel.GetSdf();
//		v1_c = voxel.sdfW;
//		voxel = map.FindVoxel(point + make_float3(1, 0, 1));
//		v2 = voxel.GetSdf();
//		v2_c = voxel.sdfW;
//		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
//		res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;
//
//		voxel = map.FindVoxel(point + make_float3(0, 1, 1));
//		v1 = voxel.GetSdf();
//		v1_c = voxel.sdfW;
//		voxel = map.FindVoxel(point + make_float3(1, 1, 1));
//		v2 = voxel.GetSdf();
//		v2_c = voxel.sdfW;
//		res2 = (1.0f - coeff.y) * res2
//				+ coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
//		res2_c = (1.0f - coeff.y) * res2_c
//				+ coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);
//
//		confidence = (1.0f - coeff.z) * res1_c + coeff.z * res2_c;
//
//		return (1.0f - coeff.z) * res1 + coeff.z * res2;
//	}
//
//	__device__
//	float readFromSDF_float_interpolated(const float3 & pos) {
//		float res1, res2, v1, v2;
//		float3 coeff;
//		coeff = pos - floor(pos);
//		int3 vpos = make_int3(pos + 0.5);
//
//		v1 = map.FindVoxel(pos + make_float3(0, 0, 0)).GetSdf();
//		v2 = map.FindVoxel(pos + make_float3(1, 0, 0)).GetSdf();
//		res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
//
//		v1 = map.FindVoxel(pos + make_float3(0, 1, 0)).GetSdf();
//		v2 = map.FindVoxel(pos + make_float3(1, 1, 0)).GetSdf();
//		res1 = (1.0f - coeff.y) * res1
//				+ coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
//
//		v1 = map.FindVoxel(pos + make_float3(0, 0, 1)).GetSdf();
//		v2 = map.FindVoxel(pos + make_float3(1, 0, 1)).GetSdf();
//		res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
//
//		v1 = map.FindVoxel(pos + make_float3(0, 1, 1)).GetSdf();
//		v2 = map.FindVoxel(pos + make_float3(1, 1, 1)).GetSdf();
//		res2 = (1.0f - coeff.y) * res2
//				+ coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
//
//		return (1.0f - coeff.z) * res1 + coeff.z * res2;
//	}
//
//	template<bool flipNormals>
//	__device__ void ComputeNormalMap() {
//		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y
//				+ blockIdx.y * blockDim.y);
//		if (x >= cols || y >= rows)
//			return;
//
//		float3 outNormal;
//		float angle;
//		float4 point = vmap.ptr(y)[x];
//		bool foundPoint = point.w > 0.0f;
//
//		computeNormalAndAngle<true, flipNormals>(foundPoint, x, y, outNormal,
//				angle);
//
//		if (foundPoint) {
//			nmap.ptr(y)[x] = outNormal;
//		}
//	}
//
//	template<bool useSmoothing, bool flipNormals>
//	__device__ inline void computeNormalAndAngle(bool & foundPoint, int &x,
//			int &y, float3 & outNormal, float& angle) {
//		if (!foundPoint)
//			return;
//
//		float4 xp1_y, xm1_y, x_yp1, x_ym1;
//
//		if (useSmoothing) {
//			if (y <= 2 || y >= rows - 3 || x <= 2 || x >= cols - 3) {
//				foundPoint = false;
//				return;
//			}
//
//			xp1_y = vmap.ptr(y)[x + 2], x_yp1 = vmap.ptr(y + 2)[x];
//			xm1_y = vmap.ptr(y)[x - 2], x_ym1 = vmap.ptr(y - 2)[x];
//		} else {
//			if (y <= 1 || y >= rows - 2 || x <= 1 || x >= cols - 2) {
//				foundPoint = false;
//				return;
//			}
//
//			xp1_y = vmap.ptr(y)[x + 1], x_yp1 = vmap.ptr(y + 1)[x];
//			xm1_y = vmap.ptr(y)[x - 1], x_ym1 = vmap.ptr(y - 1)[x];
//		}
//
//		float4 diff_x = make_float4(0.0f, 0.0f, 0.0f, 0.0f), diff_y =
//				make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//
//		bool doPlus1 = false;
//		if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)
//			doPlus1 = true;
//		else {
//			diff_x = xp1_y - xm1_y, diff_y = x_yp1 - x_ym1;
//
//			float length_diff = fmax(
//					diff_x.x * diff_x.x + diff_x.y * diff_x.y
//							+ diff_x.z * diff_x.z,
//					diff_y.x * diff_y.x + diff_y.y * diff_y.y
//							+ diff_y.z * diff_y.z);
//
//			if (length_diff * DeviceMap::VoxelSize * DeviceMap::VoxelSize
//					> (0.15f * 0.15f))
//				doPlus1 = true;
//		}
//
//		if (doPlus1) {
//			if (useSmoothing) {
//				xp1_y = vmap.ptr(y)[x + 1];
//				x_yp1 = vmap.ptr(y + 1)[x];
//				xm1_y = vmap.ptr(y)[x - 1];
//				x_ym1 = vmap.ptr(y - 2)[x];
//				diff_x = xp1_y - xm1_y;
//				diff_y = x_yp1 - x_ym1;
//			}
//
//			if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0) {
//				foundPoint = false;
//				return;
//			}
//		}
//
//		outNormal.x = -(diff_x.y * diff_y.z - diff_x.z * diff_y.y);
//		outNormal.y = -(diff_x.z * diff_y.x - diff_x.x * diff_y.z);
//		outNormal.z = -(diff_x.x * diff_y.y - diff_x.y * diff_y.x);
//
//		if (flipNormals)
//			outNormal = -outNormal;
//
//		float normScale = 1.0f
//				/ sqrt(
//						outNormal.x * outNormal.x + outNormal.y * outNormal.y
//								+ outNormal.z * outNormal.z);
//		outNormal = outNormal * normScale;
//	}
//
//	__device__ inline
//	void drawPixelNormal(uchar4 & dest, float3 & normal_obj) {
//		dest.x = (unsigned char) ((0.3f + (-normal_obj.x + 1.0f) * 0.35f)
//				* 255.0f);
//		dest.y = (unsigned char) ((0.3f + (-normal_obj.y + 1.0f) * 0.35f)
//				* 255.0f);
//		dest.z = (unsigned char) ((0.3f + (-normal_obj.z + 1.0f) * 0.35f)
//				* 255.0f);
//	}
//
//	__device__
//	void operator()() {
//		const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//		const uint y = blockIdx.y * blockDim.y + threadIdx.y;
//
//		vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
//		nmap.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));
//
//		int2 locId = make_int2(__float2int_rd((float) x / minmaximg_subsample),
//				__float2int_rd((float) y / minmaximg_subsample));
//
//		float time_min = minDepthMap.ptr(locId.y)[locId.x];
//		float time_max = maxDepthMap.ptr(locId.y)[locId.x];
//
//		if (x >= cols || y >= rows)
//			return;
//
//		if (time_min == 0 || time_min == __int_as_float(0x7fffffff))
//			return;
//		if (time_max == 0 || time_max == __int_as_float(0x7fffffff))
//			return;
//
//		time_min = max(time_min, DeviceMap::DepthMin);
//		time_max = min(time_max, DeviceMap::DepthMax);
//
//		float4 pt_out;
//
//		if (castRay(pt_out, x, y, DeviceMap::VoxelSize,
//				1 / DeviceMap::VoxelSize, time_min, time_max)) {
//			vmap.ptr(y)[x] = pt_out;
//		}
//	}
//};
//
//CUDA_KERNEL void projectAndSplitBlocksKernel(HashRayCaster hrc) {
//	hrc.projectAndSplitBlocks_device();
//}
//
//CUDA_KERNEL void fillBlocksKernel(HashRayCaster hrc) {
//	hrc.fillBlocks_device();
//}
//
//CUDA_KERNEL void hashRayCastKernel(HashRayCaster hrc) {
//	hrc();
//}
//
//CUDA_KERNEL void computeNormalAndAngleKernel(HashRayCaster hrc) {
//	hrc.ComputeNormalMap<false>();
//}
//
//template<typename T>
//CUDA_KERNEL void FillArray2DKernel(PtrStepSz<T> array, T val) {
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	if (x < array.cols && y < array.rows) {
//		array.ptr(y)[x] = val;
//	}
//}
//
//void Mapping::RenderMap(DeviceArray2D<float4> & vmap,
//						DeviceArray2D<float3> & nmap,
//						Matrix3f viewRot,
//						Matrix3f viewRotInv,
//						float3 viewTrans,
//						int num_occupied_blocks) {
//
//	if (num_occupied_blocks == 0)
//		return;
//
//	DeviceArray<uint> noTotalBlocks(1);
//	noTotalBlocks.zero();
//
//	dim3 b(8, 8);
//	dim3 g(cv::divUp(mDepthMapMin.cols(), b.x),
//		   cv::divUp(mDepthMapMin.rows(), b.y));
//
//	FillArray2DKernel<float> <<<g, b>>>(mDepthMapMin, DeviceMap::DepthMax);
//	FillArray2DKernel<float> <<<g, b>>>(mDepthMapMax, DeviceMap::DepthMin);
//
//	HashRayCaster hrc;
//	hrc.map = *this;
//	hrc.nVoxelBlocks = num_occupied_blocks;
//	hrc.fx = Frame::fx(0);
//	hrc.fy = Frame::fy(0);
//	hrc.cx = Frame::cx(0);
//	hrc.cy = Frame::cy(0);
//	hrc.cols = vmap.cols();
//	hrc.rows = vmap.rows();
//	hrc.Rot = viewRot;
//	hrc.invRot = viewRotInv;
//	hrc.trans = viewTrans;
//	hrc.renderingBlockList = mRenderingBlockList;
//	hrc.minDepthMap = mDepthMapMin;
//	hrc.maxDepthMap = mDepthMapMax;
//	hrc.nRenderingBlocks = noTotalBlocks;
//	hrc.vmap = vmap;
//	hrc.nmap = nmap;
//
//	const dim3 block(256);
//	const dim3 grid(cv::divUp(num_occupied_blocks, block.x));
//
//	projectAndSplitBlocksKernel<<<grid, block>>>(hrc);
//
//
//	uint totalBlocks;
//	noTotalBlocks.download((void*) &totalBlocks);
//	if (totalBlocks == 0 || totalBlocks >= DeviceMap::MaxRenderingBlocks)
//		return;
//
//	dim3 blocks = dim3(16, 16);
//	dim3 grids = dim3((unsigned int) ceil((float) totalBlocks / 4.0f), 4);
//
//	fillBlocksKernel<<<grids, blocks>>>(hrc);
//
//	dim3 block1(32, 8);
//	dim3 grid1(cv::divUp(vmap.cols(), block1.x),
//			cv::divUp(vmap.rows(), block1.y));
//
//	hashRayCastKernel<<<grid1, block1>>>(hrc);
//	computeNormalAndAngleKernel<<<grid1, block1>>>(hrc);
//}

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
	PtrStep<uchar3> rgb;
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

		float scale = DeviceMap::BlockSize * DeviceMap::VoxelSize;
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
				float thresh = DeviceMap::TruncateDist / 2;
				float z_near = min(DEPTH_MAX, z - thresh);
				float z_far = min(DEPTH_MAX, z + thresh);
				if (z_near >= z_far)
					return;
				float3 pt_near = UnprojectWorld(x, y, z_near)
						/ DeviceMap::VoxelSize;
				float3 pt_far = UnprojectWorld(x, y, z_far)
						/ DeviceMap::VoxelSize;
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
		uint val = 0;
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.hashEntries.size) {
			HashEntry& entry = map.hashEntries[x];
			if (entry.ptr != EntryAvailable) {
				if (CheckBlockVisibility(entry.pos)) {
					bScan = true;
					val = 1;
				}
			}
		}
		__syncthreads();
		if (bScan) {
			int offset = ComputeOffset<1024>(val, map.noVisibleBlocks);
			if (offset != -1 && offset < map.visibleEntries.size
					&& x < map.hashEntries.size)
				map.visibleEntries[offset] = map.hashEntries[x];
		}
	}

	template<bool fuse>
	DEV_FUNC void UpdateMap() {

		HashEntry& entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;
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
		float trunc_dist = DeviceMap::TruncateDist;
		float sdf = dp_scaled - pos.z;
		if (sdf >= -trunc_dist) {
			sdf = fmin(1.0f, sdf / trunc_dist);
			uchar3 color = rgb.ptr(uv.y)[uv.x];
			Voxel curr(sdf, 1, color, 1);
			if (abs(entry.ptr + idx) < map.voxelBlocks.size) {
				Voxel & prev = map.voxelBlocks[entry.ptr + idx];
				if (fuse) {
					if (prev.sdfW < 1e-7)
						curr.sdfW = 1;
					if (prev.rgbW < 1e-7)
						curr.rgbW = 1;
					prev += curr;
				} else {
					prev -= curr;
				}
			} else
				printf("%d\n", blockIdx.x);
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

void Mapping::FuseDepthAndColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, Matrix3f viewRot,
		Matrix3f viewRotInv, float3 viewTrans, float fx, float fy, float cx,
		float cy, float depthMin, float depthMax, uint& noblock) {

	int cols = depth.cols();
	int rows = depth.rows();

	HashIntegrator HI;
	HI.map = *this;
	HI.Rot = viewRot;
	HI.invRot = viewRotInv;
	HI.trans = viewTrans;
	HI.fx = fx;
	HI.fy = fy;
	HI.cx = cx;
	HI.cy = cy;
	HI.invfx = 1.0 / fx;
	HI.invfy = 1.0 / fy;
	HI.depth = depth;
	HI.rgb = color;
	HI.rows = rows;
	HI.cols = cols;
	HI.DEPTH_MAX = depthMax;
	HI.DEPTH_MIN = depthMin;

	dim3 block(32, 8);
	dim3 grid(cv::divUp(cols, block.x), cv::divUp(rows, block.y));
	createBlocksKernel<<<grid, block>>>(HI);

	dim3 block1(1024);
	dim3 grid1(cv::divUp((int)DeviceMap::NumEntries, block1.x));

	mNumVisibleEntries.zero();
	compacitifyEntriesKernel<<<grid1, block1>>>(HI);

	noblock = 0;
	mNumVisibleEntries.download((void*) &noblock);

	if (noblock == 0)
		return;

	dim3 block2(512);
	dim3 grid2(noblock);
	hashIntegrateKernal<true> <<<grid2, block2>>>(HI);

	return;
}

CUDA_KERNEL void resetHashKernel(HashEntry * hash, HashEntry * compact) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= DeviceMap::NumEntries)
		return;

	hash[idx].release();
	compact[idx].release();
}

CUDA_KERNEL void resetHeapKernel(int * heap, int * heap_counter, Voxel * voxels,
		int* entryPtr) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > DeviceMap::NumSdfBlocks)
		return;

	heap[idx] = DeviceMap::NumSdfBlocks - idx - 1;

	uint block_idx = idx * DeviceMap::BlockSize3;

	for (uint i = 0; i < DeviceMap::BlockSize3; ++i, ++block_idx) {
		voxels[block_idx].release();
	}

	if (idx == 0) {
		*heap_counter = DeviceMap::NumSdfBlocks - 1;
		*entryPtr = 1;
	}
}

CUDA_KERNEL void resetMutexKernel(int * mutex) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= DeviceMap::NumBuckets)
		return;

	mutex[idx] = EntryAvailable;
}

void Mapping::ResetDeviceMemory() {

	dim3 block(1024);
	dim3 grid;

	grid.x = cv::divUp((int) DeviceMap::NumSdfBlocks, block.x);
	resetHeapKernel<<<grid, block>>>(mMemory, mUsedMem, mVoxelBlocks,
			mEntryPtr);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	grid.x = cv::divUp((int) DeviceMap::NumEntries, block.x);
	resetHashKernel<<<grid, block>>>(mHashEntries, mVisibleEntries);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	grid.x = cv::divUp((int) DeviceMap::NumBuckets, block.x);
	resetMutexKernel<<<grid, block>>>(mBucketMutex);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	ResetKeys(*this);
	mCamTrace.clear();
}

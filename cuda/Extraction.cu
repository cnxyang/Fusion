#include "Mapping.hpp"

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

struct HashMarchingCube {

	DeviceMap map;
	float3* triangles;
	PtrStep<int> triangleTable;
	PtrSz<int> edgeTable;
	PtrSz<int3> vPos;
	uint* noBlocks;
	uint* noTriangles;
	float2* thresh;
	Matrix3f Rot, invRot;
	float3 trans;
	uchar3* color;
	float3* normals;
	int cols, rows;
	float fx, fy, cx, cy;

	DEV_FUNC void FindExistingBlocks() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		__shared__ bool scan;
		if(x == 0)
			scan = false;
		__syncthreads();
		uint val = 0;
		if (x < DeviceMap::NumEntries) {
			if (map.hashEntries[x].ptr >= 0) {
				int3 pos = map.hashEntries[x].pos * DeviceMap::BlockSize;
//				if(map.FindVoxel(pos).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(7, 0, 0)).GetSdf() < 0 ||
//			       map.FindVoxel(pos + make_int3(0, 7, 0)).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(7, 0, 7)).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(7, 7, 0)).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(7, 0, 7)).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(0, 7, 7)).GetSdf() < 0 ||
//				   map.FindVoxel(pos + make_int3(7, 7, 7)).GetSdf() < 0) {
					scan = true;
					val = 1;
//				}
			}
		}
		__syncthreads();
		if(scan) {
			int offset = ComputeOffset<1024>(val, noBlocks);
			if(offset != -1) {
				vPos[offset] = map.hashEntries[x].pos;
			}
		}
	}

	DEV_FUNC bool FindNormal(float3* n, float* sdf, int3 pos) {
		float v1, v2, v3;
		v1 = map.FindVoxel(pos + make_int3(-1, 0, 0)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(0, -1, 0)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(0, 0, -1)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[0] = make_float3(sdf[1] - v1, sdf[3] - v2, sdf[4] - v3);

		v1 = map.FindVoxel(pos + make_int3(2, 0, 0)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(1, -1, 0)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(1, 0, -1)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[1] = make_float3(v1 - sdf[0], sdf[2] - v2, sdf[5] - v3);

		v1 = map.FindVoxel(pos + make_int3(2, 1, 0)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(1, 2, 0)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(1, 1, -1)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[2] = make_float3(v1 - sdf[3], v2 - sdf[1], sdf[6] - v3);

		v1 = map.FindVoxel(pos + make_int3(-1, 1, 0)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(0, 2, 0)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(0, 1, -1)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[3] = make_float3(sdf[2] - v1, v2 - sdf[0], sdf[7] - v3);

		v1 = map.FindVoxel(pos + make_int3(-1, 0, 1)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(0, -1, 1)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(0, 0, 2)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[4] = make_float3(sdf[5] - v1, sdf[7] - v2, v3 - sdf[0]);

		v1 = map.FindVoxel(pos + make_int3(2, 0, 1)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(1, -1, 1)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(1, 0, 2)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[5] = make_float3(v1 - sdf[4], sdf[6] - v2 , v3 - sdf[1]);

		v1 = map.FindVoxel(pos + make_int3(2, 1, 1)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(1, 2, 1)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(1, 1, 2)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[6] = make_float3(v1 - sdf[7], v2 - sdf[5] , v3 - sdf[2]);

		v1 = map.FindVoxel(pos + make_int3(-1, 1, 1)).GetSdf();
		v2 = map.FindVoxel(pos + make_int3(0, 2, 1)).GetSdf();
		v3 = map.FindVoxel(pos + make_int3(0, 1, 2)).GetSdf();
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[7] = make_float3(sdf[6] - v1, v2 - sdf[4] , v3 - sdf[3]);

		return true;
	}

	DEV_FUNC bool FindPoint(float3* p, float3* n, uchar3* c, float* sdf, int3 pos) {

		int3 localPos;
		Voxel v;

		localPos = pos + make_int3(0, 0, 0);
		p[0] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[0] = v.GetSdf();
		c[0] = v.rgb;
		if (sdf[0] == 1.0 || isnan(sdf[0]))
			return false;

		localPos = pos + make_int3(1, 0, 0);
		p[1] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[1] = v.GetSdf();
		c[1] = v.rgb;
		if (sdf[1] == 1.0 || isnan(sdf[1]))
			return false;

		localPos = pos + make_int3(1, 1, 0);
		p[2] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[2] = v.GetSdf();
		c[2] = v.rgb;
		if (sdf[2] == 1.0f || isnan(sdf[2]))
			return false;

		localPos = pos + make_int3(0, 1, 0);
		p[3] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[3] = v.GetSdf();
		c[3] = v.rgb;
		if (sdf[3] == 1.0f || isnan(sdf[3]))
			return false;

		localPos = pos + make_int3(0, 0, 1);
		p[4] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[4] = v.GetSdf();
		c[4] = v.rgb;
		if (sdf[4] == 1.0f || isnan(sdf[4]))
			return false;

		localPos = pos + make_int3(1, 0, 1);
		p[5] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[5] = v.GetSdf();
		c[5] = v.rgb;
		if (sdf[5] == 1.0f || isnan(sdf[5]))
			return false;

		localPos = pos + make_int3(1, 1, 1);
		p[6] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[6] = v.GetSdf();
		c[6] = v.rgb;
		if (sdf[6] == 1.0f || isnan(sdf[6]))
			return false;

		localPos = pos + make_int3(0, 1, 1);
		p[7] = make_float3(localPos);
		v = map.FindVoxel(localPos);
		sdf[7] = v.GetSdf();
		c[7] = v.rgb;
		if (sdf[7] == 1.0f || isnan(sdf[7]))
			return false;

		if(!FindNormal(n, sdf, pos))
			return false;

		return true;
	}

	template<typename T>
	DEV_FUNC T Interp(T& p1, T& p2, float val1, float val2) {
		if(fabs(0.0f - val1) < 1e-5)
			return p1;
		if(fabs(0.0f - val2) < 1e-5)
			return p2;
		if(fabs(val1 - val2) < 1e-5)
			return p1;
		return p1 + ((0.0f - val1) / (val2 - val1)) * (p2 - p1);
	}

	DEV_FUNC int BuildVertex(float3* vertList, float3* nlist, uchar3* clist, int3 blockPos, int3 localPos)	{
		float3 points[8];
		float3 normal[8];
		uchar3 color[8];
		float sdf[8];

		if (!FindPoint(points, normal, color, sdf, blockPos + localPos))
			return -1;

		int cubeIndex = 0;
		if (sdf[0] < 0)
			cubeIndex |= 1;
		if (sdf[1] < 0)
			cubeIndex |= 2;
		if (sdf[2] < 0)
			cubeIndex |= 4;
		if (sdf[3] < 0)
			cubeIndex |= 8;
		if (sdf[4] < 0)
			cubeIndex |= 16;
		if (sdf[5] < 0)
			cubeIndex |= 32;
		if (sdf[6] < 0)
			cubeIndex |= 64;
		if (sdf[7] < 0)
			cubeIndex |= 128;

		if (edgeTable[cubeIndex] == 0)
			return -1;

		if (edgeTable[cubeIndex] & 1) {
			vertList[0] = Interp<float3>(points[0], points[1], sdf[0], sdf[1]);
//			nlist[0] = Interp<float3>(normal[0], normal[1], sdf[0], sdf[1]);
			clist[0] = Interp<uchar3>(color[0], color[1], sdf[0], sdf[1]);
		}
		if (edgeTable[cubeIndex] & 2) {
			vertList[1] = Interp<float3>(points[1], points[2], sdf[1], sdf[2]);
//			nlist[1] = Interp<float3>(normal[1], normal[2], sdf[1], sdf[2]);
			clist[1] = Interp<uchar3>(color[1], color[2], sdf[1], sdf[2]);
		}
		if (edgeTable[cubeIndex] & 4) {
			vertList[2] = Interp<float3>(points[2], points[3], sdf[2], sdf[3]);
//			nlist[2] = Interp<float3>(normal[2], normal[3], sdf[2], sdf[3]);
			clist[2] = Interp<uchar3>(color[2], color[3], sdf[2], sdf[3]);
		}
		if (edgeTable[cubeIndex] & 8) {
			vertList[3] = Interp<float3>(points[3], points[0], sdf[3], sdf[0]);
//			nlist[3] = Interp<float3>(normal[3], normal[0], sdf[3], sdf[0]);
			clist[3] = Interp<uchar3>(color[3], color[0], sdf[3], sdf[0]);
		}
		if (edgeTable[cubeIndex] & 16) {
			vertList[4] = Interp<float3>(points[4], points[5], sdf[4], sdf[5]);
//			nlist[4] = Interp<float3>(normal[4], normal[5], sdf[4], sdf[5]);
			clist[4] = Interp<uchar3>(color[4], color[5], sdf[4], sdf[5]);
		}
		if (edgeTable[cubeIndex] & 32) {
			vertList[5] = Interp<float3>(points[5], points[6], sdf[5], sdf[6]);
//			nlist[5] = Interp<float3>(normal[5], normal[6], sdf[5], sdf[6]);
			clist[5] = Interp<uchar3>(color[5], color[6], sdf[5], sdf[6]);
		}
		if (edgeTable[cubeIndex] & 64) {
			vertList[6] = Interp<float3>(points[6], points[7], sdf[6], sdf[7]);
//			nlist[6] = Interp<float3>(normal[6], normal[7], sdf[6], sdf[7]);
			clist[6] = Interp<uchar3>(color[6], color[7], sdf[6], sdf[7]);
		}
		if (edgeTable[cubeIndex] & 128) {
			vertList[7] = Interp<float3>(points[7], points[4], sdf[7], sdf[4]);
//			nlist[7] = Interp<float3>(normal[7], normal[4], sdf[7], sdf[4]);
			clist[7] = Interp<uchar3>(color[7], color[4], sdf[7], sdf[4]);
		}
		if (edgeTable[cubeIndex] & 256) {
			vertList[8] = Interp<float3>(points[0], points[4], sdf[0], sdf[4]);
//			nlist[8] = Interp<float3>(normal[0], normal[4], sdf[0], sdf[4]);
			clist[8] = Interp<uchar3>(color[0], color[4], sdf[0], sdf[4]);
		}
		if (edgeTable[cubeIndex] & 512) {
			vertList[9] = Interp<float3>(points[1], points[5], sdf[1], sdf[5]);
//			nlist[9] = Interp<float3>(normal[1], normal[5], sdf[1], sdf[5]);
			clist[9] = Interp<uchar3>(color[1], color[5], sdf[1], sdf[5]);
		}
		if (edgeTable[cubeIndex] & 1024) {
			vertList[10] = Interp<float3>(points[2], points[6], sdf[2], sdf[6]);
//			nlist[10] = Interp<float3>(normal[2], normal[6], sdf[2], sdf[6]);
			clist[10] = Interp<uchar3>(color[2], color[6], sdf[2], sdf[6]);
		}
		if (edgeTable[cubeIndex] & 2048) {
			vertList[11] = Interp<float3>(points[3], points[7], sdf[3], sdf[7]);
//			nlist[11] = Interp<float3>(normal[3], normal[7], sdf[3], sdf[7]);
			clist[11] = Interp<uchar3>(color[3], color[7], sdf[3], sdf[7]);
		}

		return cubeIndex;
	}

	DEV_FUNC void MarchingCube() {
		int x = blockIdx.y * gridDim.x + blockIdx.x;
		if(*noTriangles >= DeviceMap::MaxTriangles)
			return;

		if(x < DeviceMap::NumSdfBlocks && x < *noBlocks) {
			float3 vlist[12];
			float3 nlist[12];
			uchar3 clist[12];
			int3 blockPos = vPos[x] * DeviceMap::BlockSize;
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
			int cubeIdx = BuildVertex(vlist, nlist, clist, blockPos, localPos);
			if(cubeIdx < 0)
				return;
			for(int i = 0; triangleTable.ptr(cubeIdx)[i] != -1; i += 3) {
				int tid = atomicAdd(noTriangles, 1);
				if(tid < DeviceMap::MaxTriangles) {
					triangles[tid * 3 + 0] = vlist[triangleTable.ptr(cubeIdx)[i + 0]] * DeviceMap::VoxelSize;
					triangles[tid * 3 + 1] = vlist[triangleTable.ptr(cubeIdx)[i + 1]] * DeviceMap::VoxelSize;
					triangles[tid * 3 + 2] = vlist[triangleTable.ptr(cubeIdx)[i + 2]] * DeviceMap::VoxelSize;
//				    normals[tid * 3 + 0] = normalised(nlist[triangleTable.ptr(cubeIdx)[i + 0]]);
//					normals[tid * 3 + 1] = normalised(nlist[triangleTable.ptr(cubeIdx)[i + 1]]);
//					normals[tid * 3 + 2] = normalised(nlist[triangleTable.ptr(cubeIdx)[i + 2]]);
					color[tid * 3 + 0] = clist[triangleTable.ptr(cubeIdx)[i + 0]];
					color[tid * 3 + 1] = clist[triangleTable.ptr(cubeIdx)[i + 1]];
					color[tid * 3 + 2] = clist[triangleTable.ptr(cubeIdx)[i + 2]];
				}
			}
		}
	}
};

__global__ void FindExistingBlocks(HashMarchingCube hmc) {
	hmc.FindExistingBlocks();
}

__global__ void MarchingCube(HashMarchingCube hmc) {
	hmc.MarchingCube();
}

uint Mapping::MeshScene() {

	DeviceArray<uint> nBlocks(1);
	DeviceArray<uint> nTriangles(1);
	DeviceArray<int3> nPos(DeviceMap::NumEntries);
	nBlocks.zero();
	nTriangles.zero();

	HashMarchingCube hmc;
	hmc.map = *this;
	hmc.triangleTable = mTriTable;
	hmc.edgeTable = mEdgeTable;
	hmc.triangles = mMesh;
	hmc.noBlocks = nBlocks;
	hmc.normals = mMeshNormal;
	hmc.noTriangles = nTriangles;
	hmc.color = mColorMap;
	hmc.vPos = nPos;

	dim3 block(MaxThread);
	dim3 grid(cv::divUp((int)DeviceMap::NumEntries, block.x));
	FindExistingBlocks<<<grid, block>>>(hmc);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	uint n;
	nBlocks.download((void*)&n);
//	std::cout << n << std::endl;
	if(n == 0)
		return 0;

	block = dim3(8, 8, 8);
	grid = dim3(cv::divUp((int)n, 16), 16);
	MarchingCube<<<grid, block>>>(hmc);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	nTriangles.download((void*)&n);
	nTriangle = min(n, DeviceMap::MaxTriangles);

	mMutexMesh.lock();
	bUpdated = true;
	mMutexMesh.unlock();
	return nTriangle;
}

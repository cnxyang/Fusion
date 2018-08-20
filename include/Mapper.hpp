#ifndef MAPPER_HPP__
#define MAPPER_HPP__

#include <vector>
#include "Frame.hpp"

#define MaxThread    1024
#define TruncateDist 0.03f
#define VoxelSize	 0.005f
#define BlockSize	 8
#define BlockSize3   512
#define nBuckets	 0x1000000

#define HOST_FUNC    __host__
#define DEV_FUNC     __device__ __inline__
#define CUDA_KERNEL	 __global__

using Eigen::Vector3d;
using Eigen::Matrix4d;

struct AllocItem {
	int3 blockPos;
	int allocType;
};

struct Block2D {
	int2 upperLeft;
	int2 lowerRight;
	float2 zRange;
};

#include "thrust/device_vector.h"

#define MaxRenderingBlocks 65535 * 4

class Mapper {
public:
	Mapper();
	~Mapper();

	template<bool checkVisibility>
	void RenderFrameView(Frame& F);

	template<bool checkVisibility>
	void RenderArbitraryView(float fx, float fy, float cx, float cy, int cols,
			int rows, Matrix3f invR, float3 t);

private:

	DeviceArray<uchar4> mRenderedScene;
	DeviceArray<float4> mRenderedVertex;
	DeviceArray<float3> mRenderedNormal;

	DeviceArray<int> mBlocksNeeded;
	DeviceArray<Block2D> mBlockProjected;
	DeviceArray<int> mVisibleBlockId;
	DeviceArray<HashEntry> mHashTable;
};

Mapper::Mapper() {

}

class DMap {
public:

	const enum EntryState {
		Available = -1,
	};

	const enum AllocState {
		MainEntry = 1, Excess = 2
	};

	enum VisibleType {
		Visible = 1, Invisible = 0, DoubleCheck = -1,
	};

	DEV_FUNC uint Hash(int3& pos);DEV_FUNC HashEntry CreateEntry(int3& pos);DEV_FUNC void CreateAllocList(
			float3& pos);

public:
	PtrSz<int> EntryPtr;
	PtrSz<int> VisibleEntries;
	PtrSz<int> VisibilityList;
	PtrSz<AllocItem> AllocList;
	PtrSz<HashEntry> HashTable;
};

#endif

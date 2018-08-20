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
#define DEV_FUNC     __device__
#define CUDA_KERNEL	 __global__

using Eigen::Vector3d;
using Eigen::Matrix4d;

struct AllocItem {
	int3 blockPos;
	int allocType;
};

class DMap {
public:

	const enum EntryState {
		Available = -1,
	};

	const enum AllocState {
		MainEntry = 1,
		Excess = 2
	};

	enum VisibleType {
		Visible = 1,
		Invisible = 0,
		DoubleCheck = -1,
	};

	DEV_FUNC uint Hash(int3& pos);
	DEV_FUNC HashEntry CreateEntry(int3& pos);
	DEV_FUNC void CreateAllocList(float3& pos);

public:

	PtrSz<int> EntryPtr;
	PtrSz<int> VisibleEntries;
	PtrSz<int> VisibilityList;
	PtrSz<AllocItem> AllocList;
	PtrSz<HashEntry> HashTable;
};

class Mapper {
public:
	Mapper();
	~Mapper();
	void ResetMap();
	void SnapShot(Frame*);
	void SnapShot(Matrix4d&);
	void FuseFrame(Frame*);
	std::vector<Frame*> GetAllKFs();

private:

	DeviceArray<HashEntry> mHashTable;
	DeviceArray<AllocItem> mAllocList;
	DeviceArray<int> mVisibilityList;
};

#endif

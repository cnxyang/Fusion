#include "device_map.hpp"

DEV_FUNC HashEntry::HashEntry() :
		pos(make_int3(0x7fffffff)), ptr(EntryAvailable), offset(0) {
}

DEV_FUNC HashEntry::HashEntry(const HashEntry & other) {
	(*this) = (other);
}

DEV_FUNC HashEntry::HashEntry(int3 pos_, int ptr, int offset) :
		pos(pos_), ptr(ptr), offset(offset) {
}

DEV_FUNC void HashEntry::release() {
	pos = make_int3(0);
	offset = 0;
	ptr = EntryAvailable;
}

DEV_FUNC void HashEntry::operator=(const HashEntry & other) {
	pos = other.pos;
	ptr = other.ptr;
	offset = other.offset;
}

DEV_FUNC bool HashEntry::operator==(const int3 & pos_) const {
	return pos == pos_ && ptr != EntryAvailable;
}

DEV_FUNC bool HashEntry::operator==(const HashEntry & other) {
	return (*this) == (other.pos);
}

DEV_FUNC Voxel::Voxel() :
		sdf(0x7fff), sdfW(0), rgb(make_uchar3(0x7fffffff)), rgbW(0){
}

DEV_FUNC Voxel::Voxel(float sdf, short weight, uchar3 rgb_, short rgbW_) :
		sdfW(weight), rgb(rgb_), rgbW(rgbW_) {
	SetSdf(sdf);

}

DEV_FUNC void Voxel::release() {
	sdf = 0x7fff;
	sdfW = 0;
	rgb = make_uchar3(0x7fffffff);
	rgbW = 0;
}

DEV_FUNC float Voxel::GetSdf() const {
	return __int2float_rn(sdf) / MaxShort;
}

DEV_FUNC void Voxel::SetSdf(float val) {
	sdf = fmaxf(-MaxShort, fminf(MaxShort, __float2int_rz(val * MaxShort)));
}

DEV_FUNC void Voxel::operator+=(const Voxel & other) {

	float sdf = GetSdf() * sdfW + other.GetSdf() * other.sdfW;
	sdf /= (sdfW + other.sdfW);
	SetSdf(sdf);
	sdfW += other.sdfW;

	float3 pcolor = make_float3(rgb);
	float3 color = make_float3(other.rgb);
	float3 res =  0.2f * color + 0.8f * pcolor;
	res = fmaxf(make_float3(0.0), fminf(res, make_float3(254.5f)));
	rgb = make_uchar3(res);
	rgbW += other.rgbW;
}

DEV_FUNC void Voxel::operator-=(const Voxel & other) {
	// TODO: de-fusion method
}

DEV_FUNC void Voxel::operator=(const Voxel & other) {
	sdf = other.sdf;
	sdfW = other.sdfW;
	rgb = other.rgb;
	rgbW = other.rgbW;
}

DEV_FUNC uint DeviceMap::Hash(const int3 & pos) {
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791))
			% NumBuckets;

	if (res < 0)
		res += NumBuckets;
	return res;
}

DEV_FUNC HashEntry DeviceMap::CreateEntry(const int3 & pos,
		const int & offset) {
	int old = atomicSub(heapCounter, 1);
	if (old >= 0) {
		int ptr = heapMem[old];
		if (ptr != -1)
			return HashEntry(pos, ptr * BlockSize3, offset);
	}
	return HashEntry(pos, EntryAvailable, 0);
}

DEV_FUNC void DeviceMap::CreateBlock(const int3& blockPos) {
	int bucketId = Hash(blockPos);
	int* mutex = &bucketMutex[bucketId];
	HashEntry* e = &hashEntries[bucketId];
	HashEntry* eEmpty = nullptr;
	if (e->pos == blockPos && e->ptr != EntryAvailable)
		return;

	if (e->ptr == EntryAvailable && !eEmpty)
		eEmpty = e;

	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		if (e->pos == blockPos && e->ptr != EntryAvailable)
			return;

		if (e->ptr == EntryAvailable && !eEmpty)
			eEmpty = e;
	}

	if (eEmpty) {
		int old = atomicExch(mutex, EntryOccupied);
		if (old == EntryAvailable) {
			*eEmpty = CreateEntry(blockPos, e->offset);
			atomicExch(mutex, EntryAvailable);
		}
	} else {
		int old = atomicExch(mutex, EntryOccupied);
		if (old == EntryAvailable) {
			int offset = atomicAdd(entryPtr, 1);
			if (offset <= NumExcess) {
				eEmpty = &hashEntries[NumBuckets + offset - 1];
				*eEmpty = CreateEntry(blockPos, 0);
				e->offset = offset;
			}
			atomicExch(mutex, EntryAvailable);
		}
	}
}

DEV_FUNC bool DeviceMap::FindVoxel(const float3 & pos, Voxel & vox) {
	int3 voxel_pos = worldPosToVoxelPos(pos);
	return FindVoxel(voxel_pos, vox);
}

DEV_FUNC bool DeviceMap::FindVoxel(const int3 & pos, Voxel & vox) {
	HashEntry entry = FindEntry(voxelPosToBlockPos(pos));
	if (entry.ptr == EntryAvailable)
		return false;
	int idx = voxelPosToLocalIdx(pos);
	vox = voxelBlocks[entry.ptr + idx];
	return true;
}

DEV_FUNC Voxel DeviceMap::FindVoxel(const int3 & pos) {
	HashEntry entry = FindEntry(voxelPosToBlockPos(pos));
	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;
	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(pos)];
}

DEV_FUNC Voxel DeviceMap::FindVoxel(const float3 & pos) {
	int3 p = make_int3(pos);
	HashEntry entry = FindEntry(voxelPosToBlockPos(p));

	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;

	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(p)];
}

DEV_FUNC HashEntry DeviceMap::FindEntry(const float3 & pos) {
	int3 blockIdx = worldPosToBlockPos(pos);

	return FindEntry(blockIdx);
}

DEV_FUNC HashEntry DeviceMap::FindEntry(const int3& blockPos) {
	uint bucketId = Hash(blockPos);
	HashEntry* e = &hashEntries[bucketId];
	if (e->pos == blockPos && e->ptr != EntryAvailable)
		return *e;

	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		if (e->pos == blockPos && e->ptr != EntryAvailable)
			return *e;
	}
	return HashEntry(blockPos, EntryAvailable, 0);
}

DEV_FUNC int3 DeviceMap::worldPosToVoxelPos(float3 pos) const {
	float3 p = pos / VoxelSize;
	return make_int3(p);
}

DEV_FUNC float3 DeviceMap::worldPosToVoxelPosF(float3 pos) const {
	return pos / VoxelSize;
}

DEV_FUNC float3 DeviceMap::voxelPosToWorldPos(int3 pos) const {
	return pos * VoxelSize;
}

DEV_FUNC int3 DeviceMap::voxelPosToBlockPos(const int3 & pos) const {
	int3 voxel = pos;

	if (voxel.x < 0)
		voxel.x -= BlockSize - 1;
	if (voxel.y < 0)
		voxel.y -= BlockSize - 1;
	if (voxel.z < 0)
		voxel.z -= BlockSize - 1;

	return voxel / BlockSize;
}

DEV_FUNC int3 DeviceMap::blockPosToVoxelPos(const int3 & pos) const {
	return pos * BlockSize;
}

DEV_FUNC int3 DeviceMap::voxelPosToLocalPos(const int3 & pos) const {
	int3 local = pos % BlockSize;

	if (local.x < 0)
		local.x += BlockSize;
	if (local.y < 0)
		local.y += BlockSize;
	if (local.z < 0)
		local.z += BlockSize;

	return local;
}

DEV_FUNC int DeviceMap::localPosToLocalIdx(const int3 & pos) const {
	return pos.z * BlockSize * BlockSize + pos.y * BlockSize + pos.x;
}

DEV_FUNC int3 DeviceMap::localIdxToLocalPos(const int & idx) const {
	uint x = idx % BlockSize;
	uint y = idx % (BlockSize * BlockSize) / BlockSize;
	uint z = idx / (BlockSize * BlockSize);
	return make_int3(x, y, z);
}

DEV_FUNC int3 DeviceMap::worldPosToBlockPos(const float3 & pos) const {
	return voxelPosToBlockPos(worldPosToVoxelPos(pos));
}

DEV_FUNC float3 DeviceMap::blockPosToWorldPos(const int3 & pos) const {
	return voxelPosToWorldPos(blockPosToVoxelPos(pos));
}

DEV_FUNC int DeviceMap::voxelPosToLocalIdx(const int3 & pos) const {
	return localPosToLocalIdx(voxelPosToLocalPos(pos));
}

DEV_FUNC int KeyMap::Hash(const int3& pos) {

	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791));
	res %= KeyMap::MaxKeys;
	if (res < 0)
		res += KeyMap::MaxKeys;

	return res;
}

DEV_FUNC ORBKey* KeyMap::FindKey(const float3& pos) {

	float3 gridPos = pos / GridSize;
	int idx = Hash(make_int3(gridPos.x, gridPos.y, gridPos.z));
	int bucketIdx = idx * nBuckets;
	const float radius = 10.0f;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		ORBKey* key = &Keys[bucketIdx];
		if (key->valid && norm(key->pos - pos) <= radius) {
			return key;
		}
	}
	return nullptr;
}

DEV_FUNC ORBKey* KeyMap::FindKey(const float3& pos, int& first, int& buck) {

	first = -1;
	float3 gridPos = pos / GridSize;
	int3 p = make_int3((int) gridPos.x, (int) gridPos.y, (int) gridPos.z);
	int idx = Hash(p);
	buck = idx;
	int bucketIdx = idx * nBuckets;
	const float radius = GridSize;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		ORBKey* key = &Keys[bucketIdx];
		if (!key->valid && first == -1)
			first = bucketIdx;

		if (key->valid && norm(key->pos - pos) <= radius) {
			return key;
		}
	}
	return nullptr;
}

DEV_FUNC void KeyMap::InsertKey(ORBKey* key) {

	int first = -1;
	int buck = 0;
	ORBKey* oldKey = FindKey(key->pos, first, buck);
	if (oldKey && oldKey->valid) {
		oldKey->obs = min(oldKey->obs + 2, MaxObs);
		return;
	} else if (first != -1) {
		int lock = atomicExch(&Mutex[buck], 1);
		if (lock < 0) {
			key->obs = 1;
			ORBKey* oldkey = &Keys[first];
			memcpy((void*) oldkey, (void*) key, sizeof(ORBKey));
		}
		atomicExch(&Mutex[buck], -1);
		return;
	}
}

DEV_FUNC void KeyMap::ResetKeys(int index) {

	if (index < Mutex.size)
		Mutex[index] = -1;

	if (index < Keys.size) {
		Keys[index].valid = false;
		Keys[index].obs = 0;
	}
}

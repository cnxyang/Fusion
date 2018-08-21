#include "device_struct.hpp"

__device__ HashEntry::HashEntry() :
		pos(make_int3(0x7fffffff)), ptr(EntryAvailable), offset(0) {
}

__device__ HashEntry::HashEntry(const HashEntry & other) {
	(*this) = (other);
}

__device__ HashEntry::HashEntry(int3 pos_, int ptr, int offset) :
		pos(pos_), ptr(ptr), offset(offset) {
}

__device__
void HashEntry::release() {
	pos = make_int3(0);
	offset = 0;
	ptr = EntryAvailable;
}

__device__
void HashEntry::operator=(const HashEntry & other) {
	pos = other.pos;
	ptr = other.ptr;
	offset = other.offset;
}

__device__
bool HashEntry::operator==(const int3 & pos_) const {
	return pos == pos_ && ptr != EntryAvailable;
}

__device__
bool HashEntry::operator==(const HashEntry & other) {
	return (*this) == (other.pos);
}

__device__ Voxel::Voxel() :
		sdf((float) 0x7fffffff), rgb(make_uchar3(0, 0, 0)), sdfW(0) {
}

__device__
void Voxel::release() {
	sdf = (float) 0x7fffffff;
	sdfW = 0;
	rgb = make_uchar3(0, 0, 0);
}

__device__
void Voxel::operator+=(const Voxel & other) {

	sdf = (sdf * sdfW + other.sdf * other.sdfW) / (sdfW + other.sdfW);
	sdfW += other.sdfW;
}

__device__
void Voxel::operator-=(const Voxel & other) {
	// TODO: de-fusion method
}

__device__
void Voxel::operator=(const Voxel & other) {
	sdf = other.sdf;
	rgb = other.rgb;
	sdfW = other.sdfW;
}

__device__ uint DeviceMap::Hash(const int3 & pos) {
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791))
			% NUM_BUCKETS;

	if (res < 0)
		res += NUM_BUCKETS;
	return res;
}

__device__ HashEntry DeviceMap::createHashEntry(const int3 & pos,
		const int & offset) {
	int old = atomicSub(heapCounter, 1);
	int ptr = heapMem[old];
	if (ptr != -1)
		return HashEntry(pos, ptr * BLOCK_SIZE, offset);
	return HashEntry(pos, EntryAvailable, offset);
}

__device__ void DeviceMap::CreateBlock(const int3 & blockPos) {
	uint bucketId = Hash(blockPos);
	uint entryId = bucketId * BUCKET_SIZE;

	int firstEmptySlot = -1;
	for (uint i = 0; i < BUCKET_SIZE; ++i, ++entryId) {

		const HashEntry & curr = hashEntries[entryId];
		if (curr.pos == blockPos)
			return;
		if (firstEmptySlot == -1 && curr.ptr == EntryAvailable) {
			firstEmptySlot = entryId;
		}
	}

	const uint lastEntryIdx = (bucketId + 1) * BUCKET_SIZE - 1;
	entryId = lastEntryIdx;
	for (int i = 0; i < LINKED_LIST_SIZE; ++i) {
		HashEntry & curr = hashEntries[entryId];
		if (curr.pos == blockPos)
			return;
		if (curr.offset == 0)
			break;

		entryId = lastEntryIdx + curr.offset % (BUCKET_SIZE * NUM_BUCKETS);
	}

	if (firstEmptySlot != -1) {
		int old = atomicExch(&bucketMutex[bucketId], EntryOccupied);

		if (old != EntryOccupied) {
			HashEntry & entry = hashEntries[firstEmptySlot];
			entry = createHashEntry(blockPos, 0);
			atomicExch(&bucketMutex[bucketId], EntryAvailable);
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

		HashEntry & curr = hashEntries[entryId];

		if (curr.ptr == EntryAvailable) {
			int old = atomicExch(&bucketMutex[bucketId], EntryOccupied);
			if (old == EntryOccupied)
				return;
			HashEntry & lastEntry = hashEntries[lastEntryIdx];
			uint bucketId2 = entryId / BUCKET_SIZE;

			old = atomicExch(&bucketMutex[bucketId2], EntryOccupied);
			if (old == EntryOccupied)
				return;

			curr = createHashEntry(blockPos, lastEntry.offset);
			atomicExch(&bucketMutex[bucketId], EntryAvailable);
			atomicExch(&bucketMutex[bucketId2], EntryAvailable);
			lastEntry.offset = offset;
			hashEntries[entryId] = lastEntry;

			return;
		}
	}
	return;
}

__device__ bool DeviceMap::searchVoxel(const float3 & pos, Voxel & vox) {
	int3 voxel_pos = worldPosToVoxelPos(pos);
	return searchVoxel(voxel_pos, vox);
}

__device__ bool DeviceMap::searchVoxel(const int3 & pos, Voxel & vox) {
	HashEntry entry = searchHashEntry(voxelPosToBlockPos(pos));
	if (entry.ptr == EntryAvailable)
		return false;
	int idx = voxelPosToLocalIdx(pos);
	vox = voxelBlocks[entry.ptr + idx];
	return true;
}

__device__ Voxel DeviceMap::searchVoxel(const int3 & pos) {
	HashEntry entry = searchHashEntry(voxelPosToBlockPos(pos));
	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;
	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(pos)];
}

__device__ Voxel DeviceMap::searchVoxel(const float3 & pos) {
	int3 p = make_int3(pos);
	HashEntry entry = searchHashEntry(voxelPosToBlockPos(p));

	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;

	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(p)];
}

__device__ HashEntry DeviceMap::searchHashEntry(const float3 & pos) {
	int3 blockIdx = worldPosToBlockPos(pos);

	return searchHashEntry(blockIdx);
}

#define HANDLE_COLLISION
__device__ HashEntry DeviceMap::searchHashEntry(const int3 & pos) {
	uint bucketIdx = Hash(pos);
	uint entryIdx = bucketIdx * BUCKET_SIZE;

	HashEntry entry(pos, EntryAvailable, 0);

	for (uint i = 0; i < BUCKET_SIZE; ++i, ++entryIdx) {
		HashEntry & curr = hashEntries[entryIdx];

		if (curr == entry)
			return curr;
	}

#ifdef HANDLE_COLLISION
	const uint lastEntryIdx = (bucketIdx + 1) * BUCKET_SIZE - 1;
	entryIdx = lastEntryIdx;

	for (int i = 0; i < LINKED_LIST_SIZE; ++i) {
		HashEntry & curr = hashEntries[entryIdx];

		if (curr == entry)
			return curr;

		if (curr.offset == 0)
			break;

		entryIdx = lastEntryIdx + curr.offset % (BUCKET_SIZE * NUM_BUCKETS);
	}
#endif

	return entry;
}

__device__ int3 DeviceMap::worldPosToVoxelPos(float3 pos) const {
	float3 p = pos / VOXEL_SIZE;
	return make_int3(p);
}

__device__ float3 DeviceMap::worldPosToVoxelPosF(float3 pos) const {
	return pos / VOXEL_SIZE;
}

__device__ float3 DeviceMap::voxelPosToWorldPos(int3 pos) const {
	return pos * VOXEL_SIZE;
}

__device__ int3 DeviceMap::voxelPosToBlockPos(const int3 & pos) const {
	int3 voxel = pos;

	if (voxel.x < 0)
		voxel.x -= BLOCK_DIM - 1;
	if (voxel.y < 0)
		voxel.y -= BLOCK_DIM - 1;
	if (voxel.z < 0)
		voxel.z -= BLOCK_DIM - 1;

	return voxel / BLOCK_DIM;
}

__device__ int3 DeviceMap::blockPosToVoxelPos(const int3 & pos) const {
	return pos * BLOCK_DIM;
}

__device__ int3 DeviceMap::voxelPosToLocalPos(const int3 & pos) const {
	int3 local = pos % BLOCK_DIM;

	if (local.x < 0)
		local.x += BLOCK_DIM;
	if (local.y < 0)
		local.y += BLOCK_DIM;
	if (local.z < 0)
		local.z += BLOCK_DIM;

	return local;
}

__device__ int DeviceMap::localPosToLocalIdx(const int3 & pos) const {
	return pos.z * BLOCK_DIM * BLOCK_DIM + pos.y * BLOCK_DIM + pos.x;
}

__device__ int3 DeviceMap::localIdxToLocalPos(const int & idx) const {
	uint x = idx % BLOCK_DIM;
	uint y = idx % (BLOCK_DIM * BLOCK_DIM) / BLOCK_DIM;
	uint z = idx / (BLOCK_DIM * BLOCK_DIM);
	return make_int3(x, y, z);
}

__device__ int3 DeviceMap::worldPosToBlockPos(const float3 & pos) const {
	return voxelPosToBlockPos(worldPosToVoxelPos(pos));
}

__device__ float3 DeviceMap::blockPosToWorldPos(const int3 & pos) const {
	return voxelPosToWorldPos(blockPosToVoxelPos(pos));
}

__device__ int DeviceMap::voxelPosToLocalIdx(const int3 & pos) const {
	return localPosToLocalIdx(voxelPosToLocalPos(pos));
}

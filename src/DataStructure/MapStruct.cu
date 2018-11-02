#include "MapStruct.h"

__device__ MapState mapState;

__device__ __host__ int MapState::maxNumVoxels() const
{
	return maxNumVoxelBlocks * blockSize3;
}

__device__ __host__ float MapState::blockWidth() const
{
	return blockSize * voxelSize;
}

__device__ __host__ int MapState::maxNumMeshVertices() const
{
	return 3 * maxNumMeshTriangles;
}

__device__ __host__ float MapState::invVoxelSize() const
{
	return 1.0f / voxelSize;
}

__device__ __host__ int MapState::numExcessEntries() const
{
	return maxNumHashEntries - maxNumBuckets;
}

__device__ __host__ float MapState::truncateDistance() const
{
	return 8.0f * voxelSize;
}

__device__ __host__ float MapState::stepScale_raycast() const
{
	return 0.5 * truncateDistance() * invVoxelSize();
}

void updateMapState(MapState state)
{
	SafeCall(cudaMemcpyToSymbol(mapState, &state, sizeof(MapState)));
}

void downloadMapState(MapState& state)
{
	SafeCall(cudaMemcpyFromSymbol(&state, mapState, sizeof(MapState)));
}

__device__ HashEntry::HashEntry() :
	pos(make_int3(0)), next(-1), offset(-1)
{
}

__device__ HashEntry::HashEntry(int3 pos, int ptr, int offset) :
	pos(pos), next(ptr), offset(offset)
{
}

__device__ HashEntry::HashEntry(const HashEntry& other)
{
	pos = other.pos;
	next = other.next;
	offset = other.offset;
}

__device__ void HashEntry::release()
{
	next = -1;
}

__device__ void HashEntry::operator=(const HashEntry& other)
{
	pos = other.pos;
	next = other.next;
	offset = other.offset;
}

__device__ bool HashEntry::operator==(const int3& pos) const
{
	return (this->pos == pos);
}

__device__ bool HashEntry::operator==(const HashEntry& other) const
{
	return other.pos == pos;
}

__device__ Voxel::Voxel()
: sdf(std::nanf("0x7fffffff")), weight(0), color(make_uchar3(0))
{
}

__device__ Voxel::Voxel(float sdf, short weight, uchar3 rgb)
: sdf(sdf), weight(weight), color(rgb)
{
}

__device__ void Voxel::release()
{
	sdf = std::nanf("0x7fffffff");
	weight = 0;
	color = make_uchar3(0);
}

__device__ void Voxel::getValue(float& sdf, uchar3& color) const
{
	sdf = this->sdf;
	color = this->color;
}

__device__ void Voxel::operator=(const Voxel& other)
{
	sdf = other.sdf;
	weight = other.weight;
	color = other.color;
}

__device__ uint MapStruct::Hash(const int3 & pos) {
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791))
			% NumBuckets;

	if (res < 0)
		res += NumBuckets;
	return res;
}

__device__ HashEntry MapStruct::CreateEntry(const int3 & pos,
		const int & offset) {
	int old = atomicSub(heapCounter, 1);
	if (old >= 0) {
		int ptr = heapMem[old];
		if (ptr != -1)
			return HashEntry(pos, ptr * BlockSize3, offset);
	}
	return HashEntry(pos, EntryAvailable, 0);
}

__device__ void MapStruct::CreateBlock(const int3& blockPos) {
	int bucketId = Hash(blockPos);
	int* mutex = &bucketMutex[bucketId];
	HashEntry* e = &hashEntries[bucketId];
	HashEntry* eEmpty = nullptr;
	if (e->pos == blockPos && e->next != EntryAvailable)
		return;

	if (e->next == EntryAvailable && !eEmpty)
		eEmpty = e;

	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		if (e->pos == blockPos && e->next != EntryAvailable)
			return;

		if (e->next == EntryAvailable && !eEmpty)
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

__device__ bool MapStruct::FindVoxel(const float3 & pos, Voxel & vox) {
	int3 voxel_pos = posWorldToVoxel(pos);
	return FindVoxel(voxel_pos, vox);
}

__device__ bool MapStruct::FindVoxel(const int3 & pos, Voxel & vox) {
	HashEntry entry = FindEntry(posVoxelToBlock(pos));
	if (entry.next == EntryAvailable)
		return false;
	int idx = posVoxelToIdx(pos);
	vox = voxelBlocks[entry.next + idx];
	return true;
}

__device__ Voxel MapStruct::FindVoxel(const int3 & pos) {
	HashEntry entry = FindEntry(posVoxelToBlock(pos));
	Voxel voxel;
	if (entry.next == EntryAvailable)
		return voxel;
	return voxelBlocks[entry.next + posVoxelToIdx(pos)];
}

__device__ Voxel MapStruct::FindVoxel(const float3 & pos) {
	int3 p = make_int3(pos);
	HashEntry entry = FindEntry(posVoxelToBlock(p));

	Voxel voxel;
	if (entry.next == EntryAvailable)
		return voxel;

	return voxelBlocks[entry.next + posVoxelToIdx(p)];
}

__device__ Voxel MapStruct::FindVoxel(const float3 & pos, HashEntry & cache, bool & valid) {
	int3 p = make_int3(pos);
	int3 blockPos = posVoxelToBlock(p);
	if(blockPos == cache.pos) {
		valid = true;
		return voxelBlocks[cache.next + posVoxelToIdx(p)];
	}

	HashEntry entry = FindEntry(blockPos);
	if (entry.next == EntryAvailable) {
		valid = false;
		return Voxel();
	}

	valid = true;
	cache = entry;
	return voxelBlocks[entry.next + posVoxelToIdx(p)];
}

__device__ HashEntry MapStruct::FindEntry(const float3 & pos) {
	int3 blockIdx = posWorldToBlock(pos);

	return FindEntry(blockIdx);
}

__device__ HashEntry MapStruct::FindEntry(const int3& blockPos) {
	uint bucketId = Hash(blockPos);
	HashEntry* e = &hashEntries[bucketId];
	if (e->next != EntryAvailable && e->pos == blockPos)
		return *e;

	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		if (e->pos == blockPos && e->next != EntryAvailable)
			return *e;
	}
	return HashEntry(blockPos, EntryAvailable, 0);
}

__device__ int3 MapStruct::posWorldToVoxel(float3 pos) const {
	float3 p = pos / VoxelSize;
	return make_int3(p);
}

__device__ float3 MapStruct::posWorldToVoxelFloat(float3 pos) const {
	return pos / VoxelSize;
}

__device__ float3 MapStruct::posVoxelToWorld(int3 pos) const {
	return pos * VoxelSize;
}

__device__ int3 MapStruct::posVoxelToBlock(const int3 & pos) const {
	int3 voxel = pos;

	if (voxel.x < 0)
		voxel.x -= BlockSize - 1;
	if (voxel.y < 0)
		voxel.y -= BlockSize - 1;
	if (voxel.z < 0)
		voxel.z -= BlockSize - 1;

	return voxel / BlockSize;
}

__device__ int3 MapStruct::posBlockToVoxel(const int3 & pos) const {
	return pos * BlockSize;
}

__device__ int3 MapStruct::posVoxelToLocal(const int3 & pos) const {
	int3 local = pos % BlockSize;

	if (local.x < 0)
		local.x += BlockSize;
	if (local.y < 0)
		local.y += BlockSize;
	if (local.z < 0)
		local.z += BlockSize;

	return local;
}

__device__ int MapStruct::posLocalToIdx(const int3 & pos) const {
	return pos.z * BlockSize * BlockSize + pos.y * BlockSize + pos.x;
}

__device__ int3 MapStruct::posIdxToLocal(const int & idx) const {
	uint x = idx % BlockSize;
	uint y = idx % (BlockSize * BlockSize) / BlockSize;
	uint z = idx / (BlockSize * BlockSize);
	return make_int3(x, y, z);
}

__device__ int3 MapStruct::posWorldToBlock(const float3 & pos) const {
	return posVoxelToBlock(posWorldToVoxel(pos));
}

__device__ float3 MapStruct::posBlockToWorld(const int3 & pos) const {
	return posVoxelToWorld(posBlockToVoxel(pos));
}

__device__ int MapStruct::posVoxelToIdx(const int3 & pos) const {
	return posLocalToIdx(posVoxelToLocal(pos));
}

///////////////////////////////////////////////////////
// Implementation - Key Maps
///////////////////////////////////////////////////////
__device__ int KeyMap::Hash(const int3 & pos) {

	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % KeyMap::MaxKeys;

	if (res < 0)
		res += KeyMap::MaxKeys;

	return res;
}

__device__ SURF * KeyMap::FindKey(const float3 & pos) {

	int3 blockPos = make_int3(pos / GridSize);
	int idx = Hash(blockPos);
	int bucketIdx = idx * nBuckets;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		SURF * key = &Keys[bucketIdx];
		if (key->valid) {
			if(make_int3(key->pos / GridSize) == blockPos)
				return key;
		}
	}
	return nullptr;
}

__device__ SURF * KeyMap::FindKey(const float3 & pos, int & first,
		int & buck, int & hashIndex) {

	first = -1;
	int3 p = make_int3(pos / GridSize);
	int idx = Hash(p);
	buck = idx;
	int bucketIdx = idx * nBuckets;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		SURF * key = &Keys[bucketIdx];
		if (!key->valid && first == -1)
			first = bucketIdx;

		if (key->valid) {
			int3 tmp = make_int3(key->pos / GridSize);
			if(tmp == p) {
				hashIndex = bucketIdx;
				return key;
			}
		}
	}

	return NULL;
}

__device__ void KeyMap::InsertKey(SURF * key, int & hashIndex) {

	int buck = 0;
	int first = -1;
	SURF * oldKey = NULL;
//	if(hashIndex >= 0 && hashIndex < Keys.size) {
//		oldKey = &Keys[hashIndex];
//		if (oldKey && oldKey->valid) {
//			key->pos = oldKey->pos;
//			return;
//		}
//	}

	oldKey = FindKey(key->pos, first, buck, hashIndex);
	if (oldKey && oldKey->valid) {
		key->pos = oldKey->pos;
		return;
	}
	else if (first != -1) {

		int lock = atomicExch(&Mutex[buck], 1);
		if (lock < 0) {
			hashIndex = first;
			SURF * oldkey = &Keys[first];
			memcpy((void*) oldkey, (void*) key, sizeof(SURF));

			atomicExch(&Mutex[buck], -1);
			return;
		}
	}
}

__device__ void KeyMap::ResetKeys(int index) {

	if (index < Mutex.size)
		Mutex[index] = -1;

	if (index < Keys.size) {
		Keys[index].valid = false;
	}
}

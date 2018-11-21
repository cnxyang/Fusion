#include "DeviceFuncs.h"
#include "DeviceParallelScan.h"

struct Fusion {

	MapStruct map;
	float invfx, invfy;
	float fx, fy, cx, cy;
	float minDepth, maxDepth;
	int width, height;
	Matrix3f Rview;
	Matrix3f RviewInv;
	float3 tview;

	uint* noVisibleBlocks;

	PtrStep<float4> nmap;
	PtrStep<float> depth;
	PtrStep<uchar3> rgb;
	PtrStep<int> weight;

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
			   pt2d.x < width && pt2d.y < height &&
			   pt3d.z >= minDepth && pt3d.z <= maxDepth;
	}

	__device__ inline bool CheckBlockVisibility(const int3& pos) {
		float scale = mapState.blockWidth();
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
		if (x >= width && y >= height)
			return;

		float z = depth.ptr(y)[x];
		if (isnan(z) || z < minDepth || z > maxDepth)
			return;

		float thresh = mapState.truncateDistance() / 2;
		float z_near = min(maxDepth, z - thresh);
		float z_far = min(maxDepth, z + thresh);
		if (z_near >= z_far)
			return;

		float3 pt_near = unproject(x, y, z_near) * mapState.invVoxelSize();
		float3 pt_far = unproject(x, y, z_far) * mapState.invVoxelSize();
		float3 dir = pt_far - pt_near;

		float length = norm(dir);
		int nSteps = (int) ceil(2.0 * length);
		dir = dir / (float) (nSteps - 1);

		for (int i = 0; i < nSteps; ++i) {
			int3 blockPos = map.posVoxelToBlock(make_int3(pt_near));
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
		if (x < mapState.maxNumHashEntries) {
			HashEntry& e = map.hashEntries[x];
			if (e.next != EntryAvailable) {
				if (CheckBlockVisibility(e.pos)) {
					bScan = true;
					val = 1;
				}
			}
		}

		__syncthreads();
		if (bScan) {
			int offset = ComputeOffset<1024>(val, noVisibleBlocks);
			if (offset != -1 &&	x < mapState.maxNumHashEntries)	{
				map.visibleEntries[offset] = map.hashEntries[x];
			}
		}
	}

	__device__ inline void fuseFrameWithColour() {

		if(blockIdx.x >= mapState.maxNumHashEntries || blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry& entry = map.visibleEntries[blockIdx.x];
		if (entry.next == EntryAvailable) return;
		int3 block_pos = map.posBlockToVoxel(entry.pos);
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			float3 pos = map.posVoxelToWorld(block_pos + localPos);
			pos = RviewInv * (pos - tview);
			int2 uv = make_int2(project(pos) + make_float2(0.5, 0.5));
			if (uv.x < 0 || uv.y < 0 || uv.x >= width || uv.y >= height)
				continue;

			float dist = depth.ptr(uv.y)[uv.x];
			if (isnan(dist) || dist > maxDepth || dist < minDepth)
				continue;

			float truncateDist = mapState.truncateDistance();
			float sdf = dist - pos.z;
			if (sdf >= -truncateDist) {
				sdf = fmin(1.0f, sdf / truncateDist);
				int& w_curr = weight.ptr(uv.y)[uv.x];
				float3 val = make_float3(rgb.ptr(uv.y)[uv.x]);
				Voxel& prev = map.voxelBlocks[entry.next + map.posLocalToIdx(localPos)];
				if (prev.weight == 0)
					prev = Voxel(sdf, w_curr, make_uchar3(val));
				else {
					val = val / 255.f;
					float3 old = make_float3(prev.color) / 255.f;
					float3 res = (0.2f * val + 0.8f * old) * 255.f;
					prev.sdf = (prev.sdf * prev.weight + sdf * w_curr) / (prev.weight + w_curr);
					prev.weight = prev.weight + w_curr;
					prev.color = make_uchar3(res);
				}
			}
		}
	}

	__device__ inline void defuseFrameWithColour() {

		if(blockIdx.x >= mapState.maxNumHashEntries || blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry& entry = map.visibleEntries[blockIdx.x];
		if (entry.next == EntryAvailable) return;
		int3 block_pos = map.posBlockToVoxel(entry.pos);
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			float3 pos = map.posVoxelToWorld(block_pos + localPos);
			pos = RviewInv * (pos - tview);
			int2 uv = make_int2(project(pos) + make_float2(0.5, 0.5));
			if (uv.x < 0 || uv.y < 0 || uv.x >= width || uv.y >= height)
				continue;

			float dist = depth.ptr(uv.y)[uv.x];
			if (isnan(dist) || dist > maxDepth || dist < minDepth)
				continue;

			float truncateDist = mapState.truncateDistance();
			float sdf = dist - pos.z;
			if (sdf >= -truncateDist)
			{
				sdf = fmin(1.0f, sdf / truncateDist);
				float3 val = make_float3(rgb.ptr(uv.y)[uv.x]);
				int& w_curr = weight.ptr(uv.y)[uv.x];
				Voxel& prev = map.voxelBlocks[entry.next + map.posLocalToIdx(localPos)];
//				val = val / 255.f;
				float3 old = make_float3(prev.color);
				float3 res = (prev.weight * old - w_curr * val);
				prev.sdf = (prev.sdf * prev.weight - sdf * w_curr);
				prev.weight = prev.weight - w_curr;
				prev.color = make_uchar3(res);

				// if weight == 0
				if (prev.weight <= 0)
				{
					prev = Voxel();
				}
				else
				{
					prev.sdf /= prev.weight;
					prev.color = prev.color / prev.weight;
				}
			}
		}
	}
};

__global__ void CreateBlocksKernel(Fusion fuse) {
	fuse.CreateBlocks();
}

__global__ void FuseColorKernal(Fusion fuse) {
	fuse.fuseFrameWithColour();
}

__global__ void DefuseColorKernal(Fusion fuse) {
	fuse.defuseFrameWithColour();
}

__global__ void CheckVisibleBlockKernel(Fusion fuse) {
	fuse.CheckFullVisibility();
}

void CheckBlockVisibility(MapStruct map,
					     DeviceArray<uint> & noVisibleBlocks,
						 Matrix3f Rview,
						 Matrix3f RviewInv,
						 float3 tview,
						 int cols,
						 int rows,
						 float fx,
						 float fy,
						 float cx,
						 float cy,
						 float depthMax,
						 float depthMin,
						 uint * host_data) {

	noVisibleBlocks.clear();

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
	fuse.height = rows;
	fuse.width = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = depthMax;
	fuse.minDepth = depthMin;

	dim3 thread = dim3(1024);
	dim3 block = dim3(divUp((int) currentState.maxNumHashEntries, thread.x));

	CheckVisibleBlockKernel<<<block, thread>>>(fuse);

	host_data[0] = 0;
	noVisibleBlocks.download((void*) host_data);
	if (host_data[0] == 0)
		return;
}

void DeFuseMap(const DeviceArray2D<float> & depth,
			   const DeviceArray2D<uchar3> & color,
			   const DeviceArray2D<float4> & nmap,
			   const DeviceArray2D<int>& weight,
			   DeviceArray<uint> & noVisibleBlocks,
			   Matrix3f Rview,
			   Matrix3f RviewInv,
			   float3 tview,
			   MapStruct map,
			   float fx,
			   float fy,
			   float cx,
			   float cy,
			   float depthMax,
			   float depthMin,
			   uint * host_data) {

	int cols = depth.cols;
	int rows = depth.rows;
	noVisibleBlocks.clear();

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
	fuse.nmap = nmap;
	fuse.height = rows;
	fuse.width = cols;
	fuse.weight = weight;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = currentState.depthMax_raycast;
	fuse.minDepth = currentState.depthMin_raycast;

	dim3 thread(16, 8);
	dim3 block(divUp(cols, thread.x), divUp(rows, thread.y));

	thread = dim3(1024);
	block = dim3(divUp((int) currentState.maxNumHashEntries, thread.x));

	CheckVisibleBlockKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	host_data[0] = 0;
	noVisibleBlocks.download((void*) host_data);
	if (host_data[0] == 0)
		return;

	thread = dim3(8, 8);
	block = dim3(host_data[0]);

	DefuseColorKernal<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void FuseMapColor(const DeviceArray2D<float> & depth,
				  const DeviceArray2D<uchar3> & color,
				  const DeviceArray2D<float4> & nmap,
				  const DeviceArray2D<int>& weight,
				  DeviceArray<uint> & noVisibleBlocks,
				  Matrix3f Rview,
				  Matrix3f RviewInv,
				  float3 tview,
				  MapStruct map,
				  float fx,
				  float fy,
				  float cx,
				  float cy,
				  float depthMax,
				  float depthMin,
				  uint * host_data) {

	int cols = depth.cols;
	int rows = depth.rows;
	noVisibleBlocks.clear();

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
	fuse.nmap = nmap;
	fuse.weight = weight;
	fuse.height = rows;
	fuse.width = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = currentState.depthMax_raycast;
	fuse.minDepth = currentState.depthMin_raycast;

	dim3 thread(16, 8);
	dim3 block(divUp(cols, thread.x), divUp(rows, thread.y));

	CreateBlocksKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	thread = dim3(1024);
	block = dim3(divUp((int) currentState.maxNumHashEntries, thread.x));

	CheckVisibleBlockKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	host_data[0] = 0;
	noVisibleBlocks.download((void*) host_data);
	if (host_data[0] == 0)
		return;

	thread = dim3(8, 8);
	block = dim3(host_data[0]);

	FuseColorKernal<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void DefuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color, const DeviceArray2D<float4> & nmap,
		DeviceArray<uint> & noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, MapStruct map, float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data)
{

	int cols = depth.cols;
	int rows = depth.rows;
	noVisibleBlocks.clear();

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
	fuse.nmap = nmap;
	fuse.height = rows;
	fuse.width = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = currentState.depthMax_raycast;
	fuse.minDepth = currentState.depthMin_raycast;

	dim3 thread = dim3(1024);
	dim3 block = dim3(divUp((int) currentState.maxNumHashEntries, thread.x));

	CheckVisibleBlockKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	host_data[0] = 0;
	noVisibleBlocks.download((void*) host_data);
	if (host_data[0] == 0)
		return;

	thread = dim3(8, 8);
	block = dim3(host_data[0]);

	DefuseColorKernal<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResetHashKernel(MapStruct map)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < mapState.maxNumHashEntries) {
		map.hashEntries[x].release();
		map.visibleEntries[x].release();
	}

	if (x < mapState.maxNumBuckets) {
		map.bucketMutex[x] = EntryAvailable;
	}
}

__global__ void ResetSdfBlockKernel(MapStruct map)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < mapState.maxNumVoxelBlocks) {
		map.heapMem[x] = mapState.maxNumVoxelBlocks - x - 1;

		int locId = x * mapState.blockSize3;
		for(int i = 0; i < mapState.blockSize3; ++i, ++locId) {
			map.voxelBlocks[locId].release();
		}
	}

	if(x == 0) {
		map.heapCounter[0] = mapState.maxNumVoxelBlocks - 1;
		map.entryPtr[0] = 1;
	}
}

void ResetMap(MapStruct map) {

	dim3 thread(1024);
	dim3 block(divUp((int) currentState.maxNumHashEntries, thread.x));

	ResetHashKernel<<<block, thread>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	block = dim3(divUp((int) currentState.maxNumVoxelBlocks, thread.x));
	ResetSdfBlockKernel<<<block, thread>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResetKeyPointsKernel(KeyMap map) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	map.ResetKeys(x);
}

void ResetKeyPoints(KeyMap map) {

	dim3 thread(1024);
	dim3 block(divUp((int) KeyMap::maxEntries, thread.x));

	ResetKeyPointsKernel<<<block, thread>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct KeyFusion {

	__device__ __forceinline__ void CollectKeys() {

		__shared__ bool scan;
		if(threadIdx.x == 0)
			scan = false;
		__syncthreads();

		uint val = 0;
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if(x < map.Keys.size) {
			SURF * key = &map.Keys[x];
			if(key->valid) {
				scan = true;
				val = 1;
			}
		}
		__syncthreads();

		if(scan) {
			int offset = ComputeOffset<1024>(val, nokeys);
			if(offset > 0 && x < map.Keys.size) {
				memcpy(&keys[offset], &map.Keys[x], sizeof(SURF));
			}
		}
	}

	__device__ __forceinline__ void InsertKeys() {

		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < size)
			map.InsertKey(&keys[x], index[x]);
	}

	KeyMap map;

	uint * nokeys;

	PtrSz<SURF> keys;

	size_t size;

	PtrSz<int> index;
};

__global__ void CollectKeyPointsKernel(KeyFusion fuse) {
	fuse.CollectKeys();
}

__global__ void InsertKeyPointsKernel(KeyFusion fuse) {
	fuse.InsertKeys();
}

void CollectKeyPoints(KeyMap map, DeviceArray<SURF> & keys, DeviceArray<uint> & noKeys) {

	KeyFusion fuse;
	fuse.map = map;
	fuse.keys = keys;
	fuse.nokeys = noKeys;

	dim3 thread(1024);
	dim3 block(divUp(map.Keys.size, thread.x));

	CollectKeyPointsKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void InsertKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<int> & keyIndex, size_t size) {

	if(size == 0)
		return;

	KeyFusion fuse;

	fuse.map = map;
	fuse.keys = keys;
	fuse.size = size;
	fuse.index = keyIndex;

	dim3 thread(1024);
	dim3 block(divUp(size, thread.x));

	InsertKeyPointsKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct KFusion
{
	int cols, rows;
	float fx, fy, cx, cy;

	PtrStep<float> lastDepth;
	PtrStep<float> nextDepth;
	PtrStep<int> lastWeight;
	PtrStep<float4> nextNMap;
	PtrStep<float4> nextVMap;
	Matrix3f R;
	float3 t;

	__device__ __inline__ void operator()()
	{
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= cols || y >= rows)
			return;

		float4& v_curr = nextVMap.ptr(y)[x];
		float4& n_curr = nextNMap.ptr(y)[x];
		if(isnan(v_curr.x) || isnan(n_curr.x))
			return;

		float3 v_currlast = R * make_float3(v_curr) + t;
		int u = (int)(fx * v_currlast.x / v_currlast.z + cx);
		int v = (int)(fy * v_currlast.y / v_currlast.z + cy);
		if(u < 0 || u >= cols || v < 0 || v >= rows)
			return;

		float& d_curr = nextDepth.ptr(y)[x];
		float& d_last = lastDepth.ptr(v)[u];
		int& w_last = lastWeight.ptr(v)[u];

		if(w_last == 0)
		{
			w_last = 1;
		}

		if(!isnan(d_last))
		{
			d_last = (d_last * w_last + v_currlast.z) / (w_last + 1);
			w_last += 1;
		}
		else if(!isnan(v_currlast.z))
		{
			d_last = v_currlast.z;
			w_last = 1;
		}
	}
};

__global__ void FuseKeyFrameDepthKernel(KFusion fusion)
{
	fusion();
}

void FuseKeyFrameDepth(DeviceArray2D<float>& lastDepth,
					   DeviceArray2D<float>& nextDepth,
					   DeviceArray2D<int>& lastWeight,
					   DeviceArray2D<float4>& nextNMap,
					   DeviceArray2D<float4>& nextVMap,
					   Matrix3f R, float3 t,
					   float* K)
{
	int cols = lastDepth.cols;
	int rows = lastDepth.rows;

	KFusion fusion;
	fusion.cols = cols;
	fusion.rows = rows;
	fusion.lastDepth = lastDepth;
	fusion.lastWeight = lastWeight;
	fusion.nextDepth = nextDepth;
	fusion.nextVMap = nextVMap;
	fusion.nextNMap = nextNMap;
	fusion.R = R;
	fusion.t = t;
	fusion.fx = K[0];
	fusion.fy = K[1];
	fusion.cx = K[2];
	fusion.cy = K[3];

	dim3 thread(16, 8);
	dim3 block(divUp(cols, thread.x), divUp(rows, thread.y));

	FuseKeyFrameDepthKernel<<<block, thread>>>(fusion);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

//	cv::Mat img(480, 640, CV_32FC1);
//	lastDepth.download(img.data, img.step);
//	cv::imshow("img", img);
//	cv::waitKey(0);
}

#include "Frame.h"
#include "VoxelMap.h"
#include "Settings.h"
#include "PointCloud.h"
#include "DeviceFuncs.h"

VoxelMap::VoxelMap() :
	meshUpdated(false), hasNewKFFlag(false),
	device(0), host(0), width(0), height(0)
{
	state.zmin_raycast_ = 0.1f;
	state.zmax_raycast_ = 3.0f;
	state.zmin_update_ = 0.1f;
	state.zmax_update_ = 3.0f;

	state.voxel_size_ = 0.006f;

	state.maxNumBuckets = 1000000;
	state.maxNumHashEntries = 1500000;
	state.maxNumVoxelBlocks = 500000;
	state.maxNumRenderingBlocks = 260000;
	state.maxNumMeshTriangles = 20000000;

	update_device_map_state();
	data.numVisibleEntries.create(1);

	nBlocks.create(1);
	noTriangles.create(1);
	blockPoses.create(state.maxNumHashEntries);
	modelVertex.create(state.num_total_mesh_vertices());
	modelNormal.create(state.num_total_mesh_vertices());
	modelColor.create(state.num_total_mesh_vertices());

	edgeTable.create(256);
	vertexTable.create(256);
	triangleTable.create(16, 256);
	edgeTable.upload(edgeTableHost);
	vertexTable.upload(vertexTableHost);
	triangleTable.upload(triangleTableHost);

	zRangeMin.create(80, 60);
	zRangeMax.create(80, 60);
	noRenderingBlocks.create(1);
	renderingBlockList.create(state.maxNumRenderingBlocks);

	noKeys.create(1);
	mutexKeys.create(KeyMap::MaxKeys);
	mapKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	surfKeys.create(3000);
	mapKeyIndex.create(3000);
}

void VoxelMap::RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float4> & nmap,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy)
{
	if (CreateRenderingBlocks(device, zRangeMin, zRangeMax, depthMax, depthMin,
			renderingBlockList, noRenderingBlocks, RviewInv, tview,
			noVisibleBlocks, fx, fy, cx, cy))
	{
		Raycast(*device, vmap, nmap, zRangeMin, zRangeMax, Rview, RviewInv,
				tview, 1.0 / fx, 1.0 / fy, cx, cy);
	}
}

void VoxelMap::CreateModel()
{
	MeshScene(nBlocks, noTriangles, *device, edgeTable, vertexTable,
			triangleTable, modelNormal, modelVertex, modelColor, blockPoses);

	noTriangles.download(&noTrianglesHost);
	if (noTrianglesHost > 0) {
		meshUpdated = true;
	}
}

int VoxelMap::fuseImages(PointCloud* pc)
{
	uint no;

	FuseMapColor(pc->depth[0],
				 pc->image_raw,
				 pc->nmap[0],
				 pc->weight,
				 data.numVisibleEntries,
				 SE3toMatrix3f(pc->frame->pose()),
				 SE3toMatrix3f(pc->frame->pose().inverse()),
				 SE3toFloat3(pc->frame->pose()),
				 *device,
				 pc->frame->fx(),
				 pc->frame->fy(),
				 pc->frame->cx(),
				 pc->frame->cy(),
				 &no);

	return (int)no;
}

int VoxelMap::defuseImages(PointCloud* pc)
{
	uint no;

	DeFuseMap(pc->depth[0],
			  pc->image_raw,
			  pc->nmap[0],
			  pc->weight,
			  data.numVisibleEntries,
			  SE3toMatrix3f(pc->frame->pose()),
			  SE3toMatrix3f(pc->frame->pose().inverse()),
			  SE3toFloat3(pc->frame->pose()),
			  *device,
			  pc->frame->fx(),
			  pc->frame->fy(),
			  pc->frame->cx(),
			  pc->frame->cy(),
			  &no);

	return (int)no;
}

void VoxelMap::raycast(PointCloud* data, int numVisibleBlocks)
{
	Frame* trackingFrame = data->frame;
	uint no = numVisibleBlocks < 0 ? updateVisibility(data) : (uint) numVisibleBlocks;
	RayTrace(no,
			SE3toMatrix3f(trackingFrame->pose()),
			SE3toMatrix3f(trackingFrame->pose().inverse()),
			SE3toFloat3(trackingFrame->pose()),
			data->vmap[0],
			data->nmap[0],
			0.1f,
			3.0f,
			trackingFrame->fx(),
			trackingFrame->fy(),
			trackingFrame->cx(),
			trackingFrame->cy());
}

uint VoxelMap::updateVisibility(PointCloud* pc)
{
	uint no(0);
	Frame* trackingFrame = pc->frame;
	CheckBlockVisibility(*device,
			data.numVisibleEntries,
			SE3toMatrix3f(trackingFrame->pose()),
			SE3toMatrix3f(trackingFrame->pose().inverse()),
			SE3toFloat3(trackingFrame->pose()),
			trackingFrame->width(),
			trackingFrame->height(),
			trackingFrame->fx(),
			trackingFrame->fy(),
			trackingFrame->cx(),
			trackingFrame->cy(),
			&no);
	return no;
}


//======================== REFACOTRING ========================

VoxelMap::VoxelMap(int w, int h) :
	width(w), height(h)
{
	if (state_initialised)
	{
		allocateDeviceMap();
	}
	else
	{
		CONSOLE_ERR("MAP PARAMETERS NOT SUPPLIED!");
		return;
	}

	data.numExistedBlocks.create(1);
	data.totalNumTriangle.create(1);
	data.numVisibleEntries.create(1);
	data.numRenderingBlocks.create(1);
	data.zRangeTopdownView.create(80, 60);
	data.zRangeSyntheticView.create(80, 60);
	data.renderingBlocks.create(state.maxNumRenderingBlocks);
	data.blockPositions.create(state.maxNumHashEntries);
	data.constEdgeTable.create(256);
	data.constVertexTable.create(256);
	data.constTriangleTtable.create(16, 256);

	data.constEdgeTable.upload(edgeTableHost);
	data.constVertexTable.upload(vertexTableHost);
	data.constTriangleTtable.upload(triangleTableHost);
}

VoxelMap::~VoxelMap()
{
	releaseHostMap();
	releaseDeviceMap();

	data.numExistedBlocks.release();
	data.totalNumTriangle.release();
	data.numVisibleEntries.release();
	data.numRenderingBlocks.release();
	data.zRangeTopdownView.release();
	data.zRangeSyntheticView.release();
	data.renderingBlocks.release();
	data.blockPositions.release();
	data.constEdgeTable.release();
	data.constVertexTable.release();
	data.constTriangleTtable.release();
}

void VoxelMap::resetMapStruct()
{
	CONSOLE("Re-Initialising Device Memory.");
	ResetMap(*device);
}

void VoxelMap::copyMapToHost()
{
	if(host && device)
	{
		cudaMemcpyFromSymbol(host->entryPtr, device->entryPtr, sizeof(int));
		cudaMemcpyFromSymbol(host->heapCounter, device->heapCounter, sizeof(int));
		cudaMemcpyFromSymbol(host->bucketMutex, device->bucketMutex, sizeof(int) * state.maxNumBuckets);
		cudaMemcpyFromSymbol(host->heapMem, device->heapMem, sizeof(int) * state.maxNumVoxelBlocks);
		cudaMemcpyFromSymbol(host->hashEntries, device->hashEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMemcpyFromSymbol(host->visibleEntries, device->visibleEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMemcpyFromSymbol(host->voxelBlocks, device->voxelBlocks, sizeof(Voxel) * state.num_total_voxel());
	}
	else
		CONSOLE_ERR("COPY MAP called without an ACTIVE map.");
}

void VoxelMap::copyMapToDevice()
{
	if(host && device)
	{
		cudaMemcpyToSymbol(device->entryPtr, host->entryPtr, sizeof(int));
		cudaMemcpyToSymbol(device->heapCounter, host->heapCounter, sizeof(int));
		cudaMemcpyToSymbol(device->bucketMutex, host->bucketMutex, sizeof(int) * state.maxNumBuckets);
		cudaMemcpyToSymbol(device->heapMem, host->heapMem, sizeof(int) * state.maxNumVoxelBlocks);
		cudaMemcpyToSymbol(device->hashEntries, host->hashEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMemcpyToSymbol(device->visibleEntries, host->visibleEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMemcpyToSymbol(device->voxelBlocks, host->voxelBlocks, sizeof(Voxel) * state.num_total_voxel());
	}
	else
		CONSOLE_ERR("COPY MAP called without an ACTIVE map.");
}

void VoxelMap::allocateDeviceMap()
{
	if(!device)
	{
		device = new MapStruct();

		cudaMalloc((void**) &device->entryPtr, sizeof(int));
		cudaMalloc((void**) &device->heapCounter, sizeof(int));
		cudaMalloc((void**) &device->bucketMutex, sizeof(int) * state.maxNumBuckets);
		cudaMalloc((void**) &device->heapMem, sizeof(int) * state.maxNumVoxelBlocks);
		cudaMalloc((void**) &device->hashEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMalloc((void**) &device->visibleEntries, sizeof(HashEntry) * state.maxNumHashEntries);
		cudaMalloc((void**) &device->voxelBlocks, sizeof(Voxel) * state.num_total_voxel());

		CONSOLE("===============================");
		CONSOLE("VOXEL MAP successfully created on DEVICE.");
		CONSOLE("COUNT(HASH ENTRY) - " + std::to_string(state.maxNumHashEntries));
		CONSOLE("COUNT(HASH BUCKET) - " + std::to_string(state.maxNumBuckets));
		CONSOLE("COUNT(VOXEL BLOCK) - " + std::to_string(state.num_total_voxel()));
		CONSOLE("SIZE(VOXEL) - " + std::to_string(state.voxel_size_));
		CONSOLE("===============================");

		resetMapStruct();
	}
	else
	{
		releaseDeviceMap();
		allocateDeviceMap();
	}
}

void VoxelMap::allocateHostMap()
{
	if(!host)
	{
		host = new MapStruct();

		host->entryPtr = new int[1];
		host->heapCounter = new int[1];
		host->bucketMutex = new int[state.maxNumBuckets];
		host->heapMem = new int[state.maxNumVoxelBlocks];
		host->hashEntries = new HashEntry[state.maxNumHashEntries];
		host->visibleEntries = new HashEntry[state.maxNumHashEntries];
		host->voxelBlocks = new Voxel[state.num_total_voxel()];
	}
	else
	{
		releaseHostMap();
		allocateHostMap();
	}
}

void VoxelMap::releaseHostMap()
{
	if(host)
	{
		if(host->heapMem) delete host->heapMem;
		if(host->heapCounter) delete host->heapCounter;
		if(host->hashEntries) delete host->hashEntries;
		if(host->bucketMutex) delete host->bucketMutex;
		if(host->entryPtr) delete host->entryPtr;
		if(host->visibleEntries) delete host->visibleEntries;
		if(host->voxelBlocks) delete host->voxelBlocks;

		delete host;

		CONSOLE("HOST Map released.");
	}
}

void VoxelMap::releaseDeviceMap()
{
	if(device)
	{
		if(device->heapMem) cudaFree((void*) device->heapMem);
		if(device->heapCounter)	cudaFree((void*) device->heapCounter);
		if(device->hashEntries)	cudaFree((void*) device->hashEntries);
		if(device->bucketMutex)	cudaFree((void*) device->bucketMutex);
		if(device->entryPtr) cudaFree((void*) device->entryPtr);
		if(device->visibleEntries) cudaFree((void*) device->visibleEntries);
		if(device->voxelBlocks)	cudaFree((void*) device->voxelBlocks);

		delete device;

		CONSOLE("DEVICE Map released.");
	}
}

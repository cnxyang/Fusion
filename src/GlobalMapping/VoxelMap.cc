#include "VoxelMap.h"
#include "Settings.h"
#include "Frame.h"
#include "PointCloud.h"
#include "DeviceFuncs.h"

VoxelMap::VoxelMap() :
	meshUpdated(false), hasNewKFFlag(false),
	device(0), host(0), width(0), height(0)
{
	currentState.blockSize = 8;
	currentState.blockSize3 = 512;
	currentState.minMaxSubSample = 8;
	currentState.renderingBlockSize = 16;

	currentState.depthMin_raycast = 0.1f;
	currentState.depthMax_raycast = 3.0f;
	currentState.voxelSize = 0.006f;

	currentState.maxNumBuckets = 1000000;
	currentState.maxNumHashEntries = 1500000;
	currentState.maxNumVoxelBlocks = 700000;
	currentState.maxNumRenderingBlocks = 260000;
	currentState.maxNumMeshTriangles = 20000000;

	updateMapState();
}

void VoxelMap::allocateDeviceMemory()
{
	data.numVisibleEntries.create(1);

	allocateDeviceMap();
	nBlocks.create(1);
	noTriangles.create(1);
	modelVertex.create(currentState.maxNumMeshVertices());
	modelNormal.create(currentState.maxNumMeshVertices());
	modelColor.create(currentState.maxNumMeshVertices());
	blockPoses.create(currentState.maxNumHashEntries);

	edgeTable.create(256);
	vertexTable.create(256);
	triangleTable.create(16, 256);
	edgeTable.upload(edgeTableHost);
	vertexTable.upload(vertexTableHost);
	triangleTable.upload(triangleTableHost);

	zRangeMin.create(80, 60);
	zRangeMax.create(80, 60);
	zRangeMinEnlarged.create(160, 120);
	zRangeMaxEnlarged.create(160, 120);
	noRenderingBlocks.create(1);
	renderingBlockList.create(currentState.maxNumRenderingBlocks);

	noKeys.create(1);
	mutexKeys.create(KeyMap::MaxKeys);
	mapKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	surfKeys.create(3000);
	mapKeyIndex.create(3000);

	resetMapStruct();
}

void VoxelMap::UpdateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy,
		uint & no)
{
	CheckBlockVisibility(*device, data.numVisibleEntries, Rview, RviewInv, tview, 640,
			480, fx, fy, cx, cy, depthMax, depthMin, &no);
}

void VoxelMap::DefuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no)
{
	DefuseMapColor(depth, color, normal, data.numVisibleEntries, Rview,
			RviewInv, tview, *device, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0), currentState.depthMin_raycast,
			currentState.depthMax_raycast, &no);

}

void VoxelMap::FuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no)
{
	FuseMapColor(depth, color, normal, data.numVisibleEntries, Rview, RviewInv,
			tview, *device, Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0), currentState.depthMin_raycast,
			currentState.depthMax_raycast, &no);
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

void VoxelMap::UpdateMapKeys()
{
//	noKeys.clear();
//	CollectKeyPoints(*device, tmpKeys, noKeys);
//
//	noKeys.download(&noKeysHost);
//	if(noKeysHost != 0) {
//		hostKeys.resize(noKeysHost);
//		tmpKeys.download(hostKeys.data(), noKeysHost);
//	}
}

void VoxelMap::CreateRAM()
{
//	downloadMapState(currentState);
//
//	heapCounterRAM = new int[1];
//	hashCounterRAM = new int[1];
//	data.numVisibleEntriesRAM = new uint[1];
//	heapRAM = new int[MapStruct::NumSdfBlocks];
//	bucketMutexRAM = new int[MapStruct::NumBuckets];
//	sdfBlockRAM = new Voxel[MapStruct::NumVoxels];
//	hashEntriesRAM = new HashEntry[MapStruct::NumEntries];
//	visibleEntriesRAM = new HashEntry[MapStruct::NumEntries];
//
//	mutexKeysRAM = new int[KeyMap::MaxKeys];
//	mapKeysRAM = new SURF[KeyMap::maxEntries];
}

void VoxelMap::DownloadToRAM()
{
//	CreateRAM();

//	heapCounter.download(heapCounterRAM);
//	hashCounter.download(hashCounterRAM);
//	data.numVisibleEntries.download(data.numVisibleEntriesRAM);
//	heap.download(heapRAM);
//	bucketMutex.download(bucketMutexRAM);
//	sdfBlock.download(sdfBlockRAM);
//	hashEntries.download(hashEntriesRAM);
//	visibleEntries.download(visibleEntriesRAM);
//
//	mutexKeys.download(mutexKeysRAM);
//	mapKeys.download(mapKeysRAM);
}

void VoxelMap::UploadFromRAM()
{
//	heapCounter.upload(heapCounterRAM);
//	hashCounter.upload(hashCounterRAM);
//	data.numVisibleEntries.upload(data.numVisibleEntriesRAM);
//	heap.upload(heapRAM);
//	bucketMutex.upload(bucketMutexRAM);
//	sdfBlock.upload(sdfBlockRAM);
//	hashEntries.upload(hashEntriesRAM);
//	visibleEntries.upload(visibleEntriesRAM);
//
//	mutexKeys.upload(mutexKeysRAM);
//	mapKeys.upload(mapKeysRAM);
}

void VoxelMap::ReleaseRAM()
{
//	delete [] heapCounterRAM;
//	delete [] hashCounterRAM;
//	delete [] data.numVisibleEntriesRAM;
//	delete [] heapRAM;
//	delete [] bucketMutexRAM;
//	delete [] sdfBlockRAM;
//	delete [] hashEntriesRAM;
//	delete [] visibleEntriesRAM;
//
//	delete [] mutexKeysRAM;
//	delete [] mapKeysRAM;
}

bool VoxelMap::HasNewKF()
{
	return hasNewKFFlag;
}

void VoxelMap::resetMapStruct()
{

	CONSOLE("Re-Initialising Device Memory.");
	ResetMap(*device);
//	ResetKeyPoints(*device);

//	mapKeys.clear();
}

VoxelMap::operator KeyMap() const
{
	KeyMap map;
	map.Keys = mapKeys;
	map.Mutex = mutexKeys;
	return map;
}

VoxelMap::operator MapStruct() const
{
//	MapStruct map;
//	map.heapMem = heap;
//	map.heapCounter = heapCounter;
//	map.noVisibleBlocks = data.numVisibleEntries;
//	map.bucketMutex = bucketMutex;
//	map.hashEntries = hashEntries;
//	map.visibleEntries = visibleEntries;
//	map.voxelBlocks = sdfBlock;
//	map.entryPtr = hashCounter;
//
//	return map;
}


int VoxelMap::fuseImages(PointCloud* pc)
{
	uint no;
	Frame* trackingFrame = pc->frame;
	FuseMapColor(pc->depth_float,
			pc->image_raw,
			pc->nmap[0],
			data.numVisibleEntries,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			*device,
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy(),
			currentState.depthMax_raycast,
			currentState.depthMin_raycast,
			&no);
	return (int)no;
}

void VoxelMap::raycast(PointCloud* data, int numVisibleBlocks)
{
	Frame* trackingFrame = data->frame;
	uint no = numVisibleBlocks < 0 ? updateVisibility(data) : (uint) numVisibleBlocks;
	RayTrace(no,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			data->vmap[0],
			data->nmap[0],
			currentState.depthMin_raycast,
			currentState.depthMax_raycast,
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy());
}

uint VoxelMap::updateVisibility(PointCloud* pc)
{
	uint no(0);
	Frame* trackingFrame = pc->frame;
	CheckBlockVisibility(*device,
			data.numVisibleEntries,
			trackingFrame->GpuRotation(),
			trackingFrame->GpuInvRotation(),
			trackingFrame->GpuTranslation(),
			trackingFrame->width(),
			trackingFrame->height(),
			trackingFrame->getfx(),
			trackingFrame->getfy(),
			trackingFrame->getcx(),
			trackingFrame->getcy(),
			currentState.depthMax_raycast,
			currentState.depthMin_raycast,
			&no);
	return no;
}


//======================== REFACOTRING ========================

VoxelMap::VoxelMap(int w, int h) :
	width(w), height(h)
{
	if (stateInitialised)
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
	data.renderingBlocks.create(currentState.maxNumRenderingBlocks);
	data.blockPositions.create(currentState.maxNumHashEntries);
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

void VoxelMap::copyMapDeviceToHost()
{
	if(host && device)
	{
		cudaMemcpyFromSymbol(host->entryPtr, device->entryPtr, sizeof(int));
		cudaMemcpyFromSymbol(host->heapCounter, device->heapCounter, sizeof(int));
		cudaMemcpyFromSymbol(host->bucketMutex, device->bucketMutex, sizeof(int) * currentState.maxNumBuckets);
		cudaMemcpyFromSymbol(host->heapMem, device->heapMem, sizeof(int) * currentState.maxNumVoxelBlocks);
		cudaMemcpyFromSymbol(host->hashEntries, device->hashEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMemcpyFromSymbol(host->visibleEntries, device->visibleEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMemcpyFromSymbol(host->voxelBlocks, device->voxelBlocks, sizeof(Voxel) * currentState.maxNumVoxels());
	}
	else
		CONSOLE_ERR("COPY Map called without an ACTIVE map.");
}

void VoxelMap::copyMapHostToDevice()
{
	if(host && device)
	{
		cudaMemcpyToSymbol(device->entryPtr, host->entryPtr, sizeof(int));
		cudaMemcpyToSymbol(device->heapCounter, host->heapCounter, sizeof(int));
		cudaMemcpyToSymbol(device->bucketMutex, host->bucketMutex, sizeof(int) * currentState.maxNumBuckets);
		cudaMemcpyToSymbol(device->heapMem, host->heapMem, sizeof(int) * currentState.maxNumVoxelBlocks);
		cudaMemcpyToSymbol(device->hashEntries, host->hashEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMemcpyToSymbol(device->visibleEntries, host->visibleEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMemcpyToSymbol(device->voxelBlocks, host->voxelBlocks, sizeof(Voxel) * currentState.maxNumVoxels());
	}
	else
		CONSOLE_ERR("COPY Map called without an ACTIVE map.");
}

void VoxelMap::allocateDeviceMap()
{
	if(!device)
	{
		device = new MapStruct();

		cudaMalloc((void**) &device->entryPtr, sizeof(int));
		cudaMalloc((void**) &device->heapCounter, sizeof(int));
		cudaMalloc((void**) &device->bucketMutex, sizeof(int) * currentState.maxNumBuckets);
		cudaMalloc((void**) &device->heapMem, sizeof(int) * currentState.maxNumVoxelBlocks);
		cudaMalloc((void**) &device->hashEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMalloc((void**) &device->visibleEntries, sizeof(HashEntry) * currentState.maxNumHashEntries);
		cudaMalloc((void**) &device->voxelBlocks, sizeof(Voxel) * currentState.maxNumVoxels());

		CONSOLE("===============================");
		CONSOLE("VOXEL MAP successfully created on DEVICE.");
		CONSOLE("COUNT(HASH ENTRY) - " + std::to_string(currentState.maxNumHashEntries));
		CONSOLE("COUNT(HASH BUCKET) - " + std::to_string(currentState.maxNumBuckets));
		CONSOLE("COUNT(VOXEL BLOCK) - " + std::to_string(currentState.maxNumVoxels()));
		CONSOLE("SIZE(VOXEL) - " + std::to_string(currentState.voxelSize));
		CONSOLE("===============================");
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
		host->bucketMutex = new int[currentState.maxNumBuckets];
		host->heapMem = new int[currentState.maxNumVoxelBlocks];
		host->hashEntries = new HashEntry[currentState.maxNumHashEntries];
		host->visibleEntries = new HashEntry[currentState.maxNumHashEntries];
		host->voxelBlocks = new Voxel[currentState.maxNumVoxels()];
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

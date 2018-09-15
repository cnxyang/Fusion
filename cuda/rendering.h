#include "devmap.h"

void rayCast(DeviceMap map,
			 DeviceArray2D<float4> & vmap,
			 DeviceArray2D<float4> & nmap,
			 DeviceArray2D<float> & zRangeX,
			 DeviceArray2D<float> & zRangeY,
			 Matrix3f Rview,
			 Matrix3f RviewInv,
			 float3 tview,
			 float invfx,
			 float invfy,
			 float cx,
			 float cy);

bool createRenderingBlock(const DeviceArray<HashEntry> & visibleBlocks,
						  DeviceArray2D<float> & zRangeX,
						  DeviceArray2D<float> & zRangeY,
						  const float & depthMax,
						  const float & depthMin,
						  DeviceArray<RenderingBlock> & renderingBlockList,
						  DeviceArray<uint> & noRenderingBlocks,
						  Matrix3f RviewInv,
						  float3 tview,
						  uint noVisibleBlocks,
						  float fx,
						  float fy,
						  float cx,
						  float cy);

uint meshScene(DeviceArray<uint> & noOccupiedBlocks,
			   DeviceArray<uint> & noTotalTriangles,
			   DeviceMap map,
			   const DeviceArray<int> & edgeTable,
			   const DeviceArray<int> & noVertexTable,
			   const DeviceArray2D<int> & triangleTable,
			   DeviceArray<float3> & normal,
			   DeviceArray<float3> & vertex,
			   DeviceArray<uchar3> & color,
			   DeviceArray<int3> & extractedBlocks);

void resetDeviceMap(DeviceMap map);
void resetKeyMap(KeyMap map);

void checkBlockInFrustum(DeviceMap map,
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
						 uint * host_data);

void integrateColor(const DeviceArray2D<float> & depth,
					const DeviceArray2D<uchar3> & color,
					DeviceArray<uint> & noVisibleBlocks,
					Matrix3f Rview,
					Matrix3f RviewInv,
					float3 tview,
					DeviceMap map,
					float fx,
					float fy,
					float cx,
					float cy,
					float depthMax,
					float depthMin,
					uint * host_data);

void ResetKeys(KeyMap map);
void CollectKeys(KeyMap map, DeviceArray<ORBKey> & keys, DeviceArray<int> & index, uint& n);
void InsertKeys(KeyMap map, DeviceArray<ORBKey>& keys, DeviceArray<int> & indices);
void ProjectVisibleKeys(KeyMap map, Matrix3f RviewInv, float3 tview,
		int cols, int rows, float fx, float fy, float cx, float cy);

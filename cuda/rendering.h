#include "device_map.hpp"

void RayCast(DeviceMap map,
			 DeviceArray2D<float4> & vmap,
			 DeviceArray2D<float3> & nmap,
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
						  const DeviceArray2D<float> & zRangeX,
						  const DeviceArray2D<float> & zRangeY,
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

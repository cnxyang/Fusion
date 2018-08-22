#include "device_map.hpp"
#include <cuda_runtime_api.h>

struct Swapper {
	DeviceMap map;
	float3 cpos;

	DEV bool NeedSwapOut(const int3& pos) {

	}

	DEV void buildSwapOutList() {

	}
};

void BuildSwapOutList() {

}

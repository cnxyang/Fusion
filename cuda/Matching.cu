#include "Converter.h"
#include "DeviceArray.h"
#include "DeviceMath.h"
#include "DeviceFunc.h"
#include "DeviceStruct.h"
#include <curand_kernel.h>
#include <opencv2/cudaarithm.hpp>

__global__ void RemoveBadDescriptors_device(cv::cuda::PtrStepSz<char> src,
																  	  	  	      cv::cuda::PtrStepSz<char> dst,
																  	  	  	      PtrSz<int> index) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= index.size)
		return;

	for(int i = 0; i < 32; ++i) {
		dst.ptr(id)[i] = src.ptr(index[id])[i];
	}
}

void RemoveBadDescriptors(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, std::vector<int>& index) {
	DeviceArray<int> gpuIndex(index.size());
	gpuIndex.upload((void*)&index[0], index.size());
	dst.create(index.size(), 32, CV_8UC1);

	dim3 block(1024);
	dim3 grid(cv::divUp(index.size(), block.x));

	RemoveBadDescriptors_device<<<grid, block>>>(src, dst, gpuIndex);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void CombineDescriptors_device(cv::cuda::PtrStepSz<char> src1,
															            cv::cuda::PtrStepSz<char> src2,
															            cv::cuda::PtrStepSz<char> dst) {

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if(id < src1.rows)
		for (int i = 0; i < 32; ++i) {
			dst.ptr(id)[i] = src1.ptr(id)[i];
		}

	if(id >= src1.rows && id < src2.rows)
		for (int i = 0; i < 32; ++i) {
			dst.ptr(id)[i] = src2.ptr(id - src1.rows)[i];
		}
}

void FuseKeyPointsAndDescriptors(Frame& frame, std::vector<MapPoint>& mps,
															   cv::cuda::GpuMat& mdesc,
															   std::vector<cv::DMatch>& matches) {

//	std::cout <<"mNKP:" << frame.mNkp << std::endl;
	int* fuseFlag = new int[frame.mNkp];
	std::memset(&fuseFlag[0], 0, sizeof(int) * frame.mNkp);

//	for(int i = 0; i < frame.mNkp; ++i)
//		std::cout << fuseFlag[i];
//	std::cout << std::endl;

	for(int i = 0; i < matches.size(); ++i) {
		int queryId = matches[i].queryIdx;
		if(queryId < frame.mNkp)
			fuseFlag[queryId] = 1;
	}

	Matrix3f R = frame.mRcw;
	float3 t =  Converter::CvMatToFloat3(frame.mtcw);
	for(int i = 0; i < frame.mNkp; ++i) {
		if(fuseFlag[i] == 0) {
			MapPoint& mp = frame.mMapPoints[i];
			mp.pos = R * mp.pos + t;
			mps.push_back(mp);
		}
	}

	DeviceArray<int> Flag(frame.mNkp);
	Flag.upload((void*)fuseFlag, frame.mNkp);

	dim3 block(1024);
	dim3 grid(cv::divUp(frame.mNkp, block.x));

	cv::cuda::GpuMat descTemp(mdesc.rows + frame.mNkp, 32, CV_8UC1);
	CombineDescriptors_device<<<grid, block>>>(mdesc, frame.mDescriptors, descTemp);
	descTemp.copyTo(mdesc);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
	delete [] fuseFlag;
}

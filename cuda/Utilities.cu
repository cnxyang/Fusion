#include "DeviceFunc.h"
#include "DeviceMath.h"
#include "DeviceArray.h"

__global__ void
BackProjectPoints_device(const PtrStepSz<float> src,
										   PtrStepSz<float4> dst,
										   float depthCutoff, float invfx, float invfy,
										   float cx, float cy) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	float4 point;
	point.z = src.ptr(y)[x];
	if(!isnan(point.z) && point.z > 1e-3) {
		point.x = point.z * (x - cx) * invfx;
		point.y = point.z * (y - cy) * invfy;
		point.w = 1.0;
	}
	else
		point.x = __int_as_float(0x7fffffff);

	dst.ptr(y)[x] = point;
}

void BackProjectPoints(const DeviceArray2D<float>& src,
									   DeviceArray2D<float4>& dst, float depthCutoff,
									   float fx, float fy, float cx, float cy) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	BackProjectPoints_device<<<grid, block>>>(src, dst, depthCutoff, 1.0 / fx, 1.0 / fy, cx, cy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ComputeNormalMap_device(const PtrStepSz<float4> src, PtrStepSz<float3> dst) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	if(x == src.cols - 1 || y == src.rows - 1) {
		dst.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));
		return;
	}

	float4 vcentre = src.ptr(y)[x];
	float4 vright = src.ptr(y)[x + 1];
	float4 vdown = src.ptr(y + 1)[x];

	if(!isnan(vcentre.x) && !isnan(vright.x) && !isnan(vdown.x)) {
		dst.ptr(y)[x] = normalised(cross(vright - vcentre, vdown - vcentre));
	}
	else
		dst.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff));
}

void ComputeNormalMap(const DeviceArray2D<float4>& src, DeviceArray2D<float3>& dst) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ComputeNormalMap_device<<<grid, block>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
WarpGrayScaleImage_device(PtrStepSz<float4> src, PtrStep<uchar> gray,
						  	  	  	  	  	     Matrix3f R1, Matrix3f invR2, float3 t1, float3 t2,
						  	  	  	  	  	     float fx, float fy, float cx, float cy,
						  	  	  	  	  	     PtrStep<uchar> diff) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	diff.ptr(y)[x] = 0;

	float3 srcp = make_float3(src.ptr(y)[x]);
	if(isnan(srcp.x) || srcp.z < 1e-6)
		return;
	float3 dst = R1 * srcp + t1;
	dst = invR2 * (dst - t2);

	int u = __float2int_rd(fx * dst.x / dst.z + cx + 0.5);
	int v = __float2int_rd(fy * dst.y / dst.z + cy + 0.5);
	if(u >= 0 && v >= 0 && u < src.cols && v < src.rows)
		diff.ptr(y)[x] = gray.ptr(v)[u];
}

void WarpGrayScaleImage(const Frame& frame1, const Frame& frame2,
										    DeviceArray2D<uchar>& diff) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(diff.cols(), block.x), cv::divUp(diff.rows(), block.y));

//	Matrix3f
//	float3 t1 = Converter::CvMatToFloat3(frame1.mtcw);
//	float3 t2 = Converter::CvMatToFloat3(frame2.mtcw);

	const int pyrlvl = 0;

	WarpGrayScaleImage_device<<<grid, block>>>(frame1.mVMap[pyrlvl], frame2.mGray[pyrlvl],
											   frame1.Rot_gpu(), frame2.RotInv_gpu(),
											   frame1.Trans_gpu(), frame2.Trans_gpu(),
											   Frame::fx(pyrlvl), Frame::fy(pyrlvl),
											   Frame::cx(pyrlvl), Frame::cy(pyrlvl), diff);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
ComputeResidualImage_device(PtrStepSz<uchar> src,
												     PtrStep<uchar> dst,
												     PtrStep<uchar> residual) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src.cols || y >= src.rows)
		return;

	residual.ptr(y)[x] = abs(src.ptr(y)[x] - dst.ptr(y)[x]);
}

void ComputeResidualImage(const DeviceArray2D<uchar>& src,
										        DeviceArray2D<uchar>& residual,
										        const Frame& frame) {
	dim3 block(8, 8);
	dim3 grid(cv::divUp(residual.cols(), block.x), cv::divUp(residual.rows(), block.y));

	const int pyrlvl = 0;

	ComputeResidualImage_device<<<grid, block>>>(src, frame.mGray[pyrlvl], residual);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void
renderImageKernel(const PtrStep<float4> vmap,
								  const PtrStep<float3> nmap,
								  const float3 light_pose,
								  PtrStepSz<uchar4> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= dst.cols || y >= dst.rows)
		return;

	float3 color;
	float3 p = make_float3(vmap.ptr(y)[x]);
	if (isnan(p.x)) {
		const float3 bgr1 = make_float3(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
		const float3 bgr2 = make_float3(236.f / 255.f, 120.f / 255.f, 120.f / 255.f);

		float w = static_cast<float>(y) / dst.rows;
		color = bgr1 * (1 - w) + bgr2 * w;
	}
	else	{
		float3 P = p;
		float3 N = nmap.ptr(y)[x];

		const float Ka = 0.3f;  //ambient coeff
		const float Kd = 0.5f;  //diffuse coeff
		const float Ks = 0.2f;  //specular coeff
		const float n = 20.f;  //specular power

		const float Ax = 1.f;   //ambient color,  can be RGB
		const float Dx = 1.f;   //diffuse color,  can be RGB
		const float Sx = 1.f;   //specular color, can be RGB
		const float Lx = 1.f;   //light color

		float3 L = normalised(light_pose - P);
		float3 V = normalised(make_float3(0.f, 0.f, 0.f) - P);
		float3 R = normalised(2 * N * (N * L) - L);

		float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (N * L))
			     + Lx * Ks * Sx * __powf(fmax(0.f, (R * V)), n);
		color = make_float3(Ix, Ix, Ix);
	}

	uchar4 out;
	out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
	out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
	out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
	out.w = 0;
	dst.ptr(y)[x] = out;
}

void renderImage(const DeviceArray2D<float4>& points,
		         	 	 	   const DeviceArray2D<float3>& normals,
		         	 	 	   const float3 & light_pose,
		         	 	 	   DeviceArray2D<uchar4>& image)
{
	dim3 block(32, 8);
	dim3 grid(cv::divUp(points.cols(), block.x), cv::divUp(points.rows(), block.y));

	renderImageKernel<<<grid, block>>>(points, normals, light_pose, image);
	SafeCall(cudaGetLastError());
}

__global__ void
ProjectToDepth_device(const PtrStep<float4> vmap_src, PtrStepSz<float> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float z = vmap_src.ptr(y)[x].z;

    dst.ptr(y)[x] = isnan(z) || z <= 1e-3 ? __int_as_float(0x7fffffff) : z;
}

void ProjectToDepth(const DeviceArray2D<float4>& src, DeviceArray2D<float>& dst)
{
    dim3 block (32, 8);
    dim3 grid (cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

    ProjectToDepth_device<<<grid, block>>>(src, dst);
    SafeCall(cudaGetLastError());
};

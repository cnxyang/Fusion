#include "mathlib.h"
#include "cuarray.h"
#include "cufunc.h"

__global__ void BackProjectPointsDevice(const PtrStepSz<float> src,
		PtrStepSz<float4> dst, float depthCutoff, float invfx, float invfy,
		float cx, float cy) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	float4 point;

	point.z = src.ptr(y)[x];
	if (!isnan(point.z) && point.z > 1e-3) {
		point.x = point.z * (x - cx) * invfx;
		point.y = point.z * (y - cy) * invfy;
		point.w = 1.0;
	}
	else {
		dst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	}

	dst.ptr(y)[x] = point;
}

void BackProjectPoints(const DeviceArray2D<float>& src,
		DeviceArray2D<float4>& dst, float depthCutoff, float fx, float fy,
		float cx, float cy) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	BackProjectPointsDevice<<<grid, block>>>(src, dst, depthCutoff, 1.0 / fx,
			1.0 / fy, cx, cy);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ComputeNormalMapDevice(const PtrStepSz<float4> src,
		PtrStepSz<float4> dst) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	if (x == src.cols - 1 || y == src.rows - 1) {
		dst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		return;
	}

	float4 vcentre = src.ptr(y)[x];
	float4 vright = src.ptr(y)[x + 1];
	float4 vdown = src.ptr(y + 1)[x];

	if (!isnan(vcentre.x) && !isnan(vright.x) && !isnan(vdown.x)) {
		dst.ptr(y)[x] = normalised(cross(vright - vcentre, vdown - vcentre));
	} else
		dst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
}

void ComputeNormalMap(const DeviceArray2D<float4>& src,
		DeviceArray2D<float4>& dst) {

	dim3 block(8, 8);
	dim3 grid(cv::divUp(src.cols(), block.x), cv::divUp(src.rows(), block.y));

	ComputeNormalMapDevice<<<grid, block>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void forwardProjectKernel(PtrStepSz<float4> src_vmap,
								     PtrStep<float4> src_nmap,
								     PtrStep<float4> dst_vmap,
								     PtrStep<float4> dst_nmap,
								     Matrix3f KRKinv, float3 Kt,
								     float fx, float fy,
								     float cx, float cy,
								     int cols, int rows) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= src_vmap.cols || y >= src_vmap.rows)
		return;

	float3 pixel = make_float3(x, y, 1.f);
	pixel = KRKinv * pixel + Kt;
	int u = __float2int_rd(pixel.x / pixel.z * fx + cx + 0.5);
	int v = __float2int_rd(pixel.y / pixel.z * fy + cy + 0.5);
	if(u < 0 || v < 0 || u >= cols || v >= rows) {
		dst_vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		dst_nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		return;
	}

	dst_vmap.ptr(y)[x] = src_vmap.ptr(v)[u];
	dst_nmap.ptr(y)[x] = src_nmap.ptr(v)[u];
}

void forwardProjection(const DeviceArray2D<float4> & vsrc,
					   const DeviceArray2D<float4> & nsrc,
					   DeviceArray2D<float4> & vdst,
					   DeviceArray2D<float4> & ndst,
					   Matrix3f KRKinv, float3 Kt,
					   float fx, float fy,
					   float cx, float cy) {

	dim3 thread(16, 8);
	dim3 block(cv::divUp(vsrc.cols(), thread.x), cv::divUp(vsrc.rows(), thread.y));

//	forwardProjectKernel<<<block, thread>>>(vsrc, nsrc, vdst, ndst, Rcurr,
//			tcurr, RlastInv, tlast, fx, fy, cx, cy);
}

__global__ void RenderImageDevice(const PtrStep<float4> vmap,
								  const PtrStep<float4> nmap,
								  const float3 lightPose,
								  PtrStepSz<uchar4> dst) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= dst.cols || y >= dst.rows)
		return;

	float3 color;
	float3 p = make_float3(vmap.ptr(y)[x]);
	if (isnan(p.x)) {
		const float3 bgr1 = make_float3(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
		const float3 bgr2 = make_float3(236.f / 255.f, 120.f / 255.f,
				120.f / 255.f);

		float w = static_cast<float>(y) / dst.rows;
		color = bgr1 * (1 - w) + bgr2 * w;
	} else {
		float3 P = p;
		float3 N = make_float3(nmap.ptr(y)[x]);

		const float Ka = 0.3f;  //ambient coeff
		const float Kd = 0.5f;  //diffuse coeff
		const float Ks = 0.2f;  //specular coeff
		const float n = 20.f;  //specular power

		const float Ax = 1.f;   //ambient color,  can be RGB
		const float Dx = 1.f;   //diffuse color,  can be RGB
		const float Sx = 1.f;   //specular color, can be RGB
		const float Lx = 1.f;   //light color

		float3 L = normalised(lightPose - P);
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
	out.w = 255.0;
	dst.ptr(y)[x] = out;
}

void RenderImage(const DeviceArray2D<float4> & points,
				 const DeviceArray2D<float4> & normals,
				 const float3 light_pose,
				 DeviceArray2D<uchar4> & image) {

	dim3 block(8, 4);
	dim3 grid(cv::divUp(points.cols(), block.x),
			  cv::divUp(points.rows(), block.y));

	RenderImageDevice<<<grid, block>>>(points, normals, light_pose, image);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

__global__ void depthToImageKernel(PtrStepSz<float> depth, PtrStepSz<uchar4> image) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= image.cols || y >= image.rows)
		return;

	float dp = depth.ptr(y)[x] / 20.0;
	int intdp = __float2int_rd(dp * 255);
	intdp = intdp > 255 ? 255 : intdp;
	image.ptr(y)[x] = make_uchar4(intdp, intdp, intdp, 255);
}

void depthToImage(const DeviceArray2D<float> & depth,
				  DeviceArray2D<uchar4> & image) {
	dim3 block(32, 8);
	dim3 grid(cv::divUp(image.cols(), block.x),
			  cv::divUp(image.rows(), block.y));

	depthToImageKernel<<<grid, block>>>(depth, image);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

__global__ void rgbImageToRgbaKernel(PtrStepSz<uchar3> image, PtrStepSz<uchar4> rgba) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= image.cols || y >= image.rows)
		return;

	uchar3 rgb = image.ptr(y)[x];
	rgba.ptr(y)[x] = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
}

void rgbImageToRgba(const DeviceArray2D<uchar3> & image,
				    DeviceArray2D<uchar4> & rgba) {
	dim3 block(32, 8);
	dim3 grid(cv::divUp(image.cols(), block.x),
			  cv::divUp(image.rows(), block.y));

	rgbImageToRgbaKernel<<<grid, block>>>(image, rgba);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

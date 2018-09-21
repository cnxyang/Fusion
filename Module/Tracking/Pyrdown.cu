#include "Reduction.h"

__constant__ float sigSpace = 0.5 / (4 * 4);
__constant__ float sigRange = 0.5 / (0.5 * 0.5);
__global__ void FilterDepthKernel(const PtrStepSz<unsigned short> depth,
		PtrStep<float> filteredDepth, float depthScaleInv) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= depth.cols || y >= depth.rows)
		return;

    float center = depth.ptr(y)[x] * depthScaleInv;
	if(isnan(center)) {
		filteredDepth.ptr(y)[x] = center;
		return;
	}

    int R = 2;
    int D = R * 2 + 1;
    int tx = min (x - D/2 + D, depth.cols - 1);
    int ty = min (y - D/2 + D, depth.rows - 1);

    float sum1 = 0;
    float sum2 = 0;
    for (int cy = max(y - D / 2, 0); cy < ty; ++cy) {
		for (int cx = max(x - D / 2, 0); cx < tx; ++cx) {
			float val = depth.ptr(cy)[cx] * depthScaleInv;
			float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			float color2 = (center - val) * (center - val);
			float weight = exp(-(space2 * sigSpace + color2 * sigRange));
			sum1 += val * weight;
			sum2 += weight;
		}
    }

    filteredDepth.ptr(y)[x] = sum1 / sum2;
}

void FilterDepth(const DeviceArray2D<unsigned short> & depth,
		DeviceArray2D<float> & filteredDepth, float depthScale) {

	dim3 thread(8, 8);
	dim3 block(DivUp(depth.cols, thread.x), DivUp(depth.rows, thread.y));

	FilterDepthKernel<<<block, thread>>>(depth, filteredDepth, 1.0 / depthScale);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ComputeVMapKernel(const PtrStepSz<float> depth,
		PtrStep<float4> vmap, float invfx, float invfy, float cx, float cy,
		float depthCutoff) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= depth.cols || y >= depth.rows)
		return;

	float4 v;
	v.z = depth.ptr(y)[x];
	if(!isnan(v.z) && v.z > 0.1 && v.z < depthCutoff) {
		v.x = v.z * (x - cx) * invfx;
		v.y = v.z * (y - cy) * invfy;
		v.w = 1.0;
	}
	else
		v.x = __int_as_float(0x7fffffff);

	vmap.ptr(y)[x] = v;
}

void ComputeVMap(const DeviceArray2D<float> & depth,
		DeviceArray2D<float4> & vmap, float fx, float fy, float cx, float cy,
		float depthCutoff) {

	dim3 thread(8, 8);
	dim3 block(DivUp(depth.cols, thread.x), DivUp(depth.rows, thread.y));

	ComputeVMapKernel<<<block, thread>>>(depth, vmap, 1.0 / fx, 1.0 / fy, cx, cy, depthCutoff);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ComputeNMapKernel(const PtrStepSz<float4> vmap,
		PtrStepSz<float4> nmap) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= vmap.cols || y >= vmap.rows)
		return;

	if (x == vmap.cols - 1 || y == vmap.rows - 1) {
		nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
		return;
	}

	float4 vcentre = vmap.ptr(y)[x];
	float4 vright = vmap.ptr(y)[x + 1];
	float4 vdown = vmap.ptr(y + 1)[x];

	if (!isnan(vcentre.x) && !isnan(vright.x) && !isnan(vdown.x)) {
		nmap.ptr(y)[x] = normalised(cross(vright - vcentre, vdown - vcentre));
	} else
		nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
}

void ComputeNMap(const DeviceArray2D<float4> & vmap, DeviceArray2D<float4> & nmap) {

	dim3 block(8, 8);
	dim3 grid(DivUp(vmap.cols, block.x), DivUp(vmap.rows, block.y));

	ComputeNMapKernel<<<grid, block>>>(vmap, nmap);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__constant__ float gaussKernel[25] = {
		1, 4, 6, 4, 1, 4,
		16, 24, 16, 4, 6,
		24, 36, 24, 6, 4,
		16, 24, 16, 4, 1,
		4, 6, 4, 1
};
template<class T, class U> __global__
void PyrDownGaussKernel(const PtrStepSz<T> src, PtrStepSz<U> dst) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dst.cols || y >= dst.rows)
		return;

	const int D = 5;
	float center = src.ptr(2 * y)[2 * x];
	int tx = min(2 * x - D / 2 + D, src.cols - 1);
	int ty = min(2 * y - D / 2 + D, src.rows - 1);
	int cy = max(0, 2 * y - D / 2);
	float sum = 0;
	int count = 0;
	for (; cy < ty; ++cy) {
		for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
			if (!isnan((float) src.ptr(cy)[cx])) {
				sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
				count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
			}
		}
	}

	dst.ptr(y)[x] = (U) (sum / (float) count);
}

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst) {

	dim3 thread(8, 8);
	dim3 block(DivUp(dst.cols, thread.x), DivUp(dst.rows, thread.y));

	PyrDownGaussKernel<float, float> <<<block, thread>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void PyrDownGauss(const DeviceArray2D<unsigned char> & src,
		DeviceArray2D<unsigned char> & dst) {

	dim3 thread(8, 8);
	dim3 block(DivUp(src.cols, thread.x), DivUp(src.rows, thread.y));

	PyrDownGaussKernel<uchar, uchar> <<<block, thread>>>(src, dst);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ImageToIntensityKernel(PtrStepSz<uchar3> src, PtrStep<unsigned char> dst) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= src.cols || y >= src.rows)
		return;

	uchar3 val = src.ptr(y)[x];
	int value = (float) val.x * 0.2126f + (float) val.y * 0.7152f + (float) val.z * 0.0722f;
	dst.ptr(y)[x] = value;
}

void ImageToIntensity(const DeviceArray2D<uchar3> & rgb, DeviceArray2D<unsigned char> & image) {

	dim3 thread(8, 8);
	dim3 block(DivUp(image.cols, thread.x), DivUp(image.rows, thread.y));

	ImageToIntensityKernel<<<block, thread>>>(rgb, image);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResizeMapKernel(const PtrStepSz<float4> vsrc, const PtrStep<float4> nsrc,
								PtrStepSz<float4> vdst,	PtrStep<float4> ndst) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= vsrc.cols || y >= vsrc.rows)
		return;

	float4 v00 = vsrc.ptr(y * 2 + 0)[x * 2 + 0];
	float4 v01 = vsrc.ptr(y * 2 + 0)[x * 2 + 1];
	float4 v10 = vsrc.ptr(y * 2 + 1)[x * 2 + 0];
	float4 v11 = vsrc.ptr(y * 2 + 1)[x * 2 + 1];
	float4 n00 = nsrc.ptr(y * 2 + 0)[x * 2 + 0];
	float4 n01 = nsrc.ptr(y * 2 + 0)[x * 2 + 1];
	float4 n10 = nsrc.ptr(y * 2 + 1)[x * 2 + 0];
	float4 n11 = nsrc.ptr(y * 2 + 1)[x * 2 + 1];

	if (isnan(v00.x) || isnan(v01.x) || isnan(v10.x) || isnan(v11.x)) {
		vdst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	} else {
		vdst.ptr(y)[x] = (v00 + v01 + v10 + v11) / 4;
	}

	if (isnan(n00.x) || isnan(n01.x) || isnan(n10.x) || isnan(n11.x)) {
		ndst.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
	} else {
		ndst.ptr(y)[x] = normalised((n00 + n01 + n10 + n11) / 4);
	}
}

void ResizeMap(const DeviceArray2D<float4>& vsrc, const DeviceArray2D<float4>& nsrc,
			   DeviceArray2D<float4>& vdst, DeviceArray2D<float4>& ndst) {

	dim3 thread(8, 8);
	dim3 block(DivUp(vdst.cols, thread.x), DivUp(vdst.rows, thread.y));

	ResizeMapKernel<<<block, thread>>>(vsrc, nsrc, vdst, ndst);

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
	dim3 grid(cv::divUp(points.cols, block.x),
			  cv::divUp(points.rows, block.y));

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

void DepthToImage(const DeviceArray2D<float> & depth,
				  DeviceArray2D<uchar4> & image) {
	dim3 block(32, 8);
	dim3 grid(cv::divUp(image.cols, block.x),
			  cv::divUp(image.rows, block.y));

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

void RgbImageToRgba(const DeviceArray2D<uchar3> & image,
				    DeviceArray2D<uchar4> & rgba) {
	dim3 block(32, 8);
	dim3 grid(cv::divUp(image.cols, block.x),
			  cv::divUp(image.rows, block.y));

	rgbImageToRgbaKernel<<<grid, block>>>(image, rgba);

	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());
}

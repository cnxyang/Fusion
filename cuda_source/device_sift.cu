#include "device_sift.cuh"
#include "device_array.hpp"

#include <cstdio>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem==NULL) {
    SafeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    SafeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL)
    SafeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
}

double CudaImage::Download()
{
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL)
    SafeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif

}

double CudaImage::Readback()
{
  int p = sizeof(float)*pitch;
  SafeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif

}

double CudaImage::InitTexture()
{
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  SafeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif

}

double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  if (host)
    SafeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    SafeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  SafeCall(cudaThreadSynchronize());
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif

}

#define NUM_SCALES      5

// Scale down thread block width
#define SCALEDOWN_W   160

// Scale down thread block height
#define SCALEDOWN_H    16

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H      8

// Find point thread block width
#define MINMAX_W      126

// Find point thread block height
#define MINMAX_H        4

// Laplace thread block width
#define LAPLACE_W      56

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      56
#define LOWPASS_H      16
#define LOWPASS_R       4

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ float d_Threshold[2];
__constant__ float d_Scales[8], d_Factor;
__constant__ float d_EdgeLimit;
__constant__ int d_MaxNumPoints;

__device__ unsigned int d_PointCounter[1];
__constant__ float d_Kernel1[5];
__constant__ float d_Kernel2[12*16];

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter and subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W+4];
  __shared__ float brow[5*(SCALEDOWN_W/2)];
  __shared__ int yRead[SCALEDOWN_H+4];
  __shared__ int yWrite[SCALEDOWN_H+4];
  #define dx2 (SCALEDOWN_W/2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0*dx2;
  const int tx1 = tx + 1*dx2;
  const int tx2 = tx + 2*dx2;
  const int tx3 = tx + 3*dx2;
  const int tx4 = tx + 4*dx2;
  const int xStart = blockIdx.x*SCALEDOWN_W;
  const int yStart = blockIdx.y*SCALEDOWN_H;
  const int xWrite = xStart/2 + tx;
  const float *k = d_Kernel1;
  if (tx<SCALEDOWN_H+4) {
    int y = yStart + tx - 1;
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = y*pitch;
    yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - 2;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);
  for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
    inrow[tx] = d_Data[yRead[dy+0] + xRead];
    __syncthreads();
    if (tx<dx2)
      brow[tx0] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
    __syncthreads();
    if (tx<dx2 && dy>=4 && !(dy&1))
      d_Result[yWrite[dy+0] + xWrite] = k[2]*brow[tx2] + k[0]*(brow[tx0]+brow[tx4]) + k[1]*(brow[tx1]+brow[tx3]);
    if (dy<(SCALEDOWN_H+3)) {
      inrow[tx] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx1] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=3 && (dy&1))
	d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]);
    }
    if (dy<(SCALEDOWN_H+2)) {
      inrow[tx] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=2 && !(dy&1))
	d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]);
    }
    if (dy<(SCALEDOWN_H+1)) {
      inrow[tx] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=1 && (dy&1))
	d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]);
    }
    if (dy<SCALEDOWN_H) {
      inrow[tx] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && !(dy&1))
	d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]);
    }
    __syncthreads();
  }
}

__global__ void ScaleUp(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  #define BW (SCALEUP_W/2 + 2)
  #define BH (SCALEUP_H/2 + 2)
  __shared__ float buffer[BW*BH];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  if (tx<BW && ty<BH) {
    int x = min(max(blockIdx.x*(SCALEUP_W/2) + tx - 1, 0), width-1);
    int y = min(max(blockIdx.y*(SCALEUP_H/2) + ty - 1, 0), height-1);
    buffer[ty*BW + tx] = d_Data[y*pitch + x];
  }
  __syncthreads();
  int x = blockIdx.x*SCALEUP_W + tx;
  int y = blockIdx.y*SCALEUP_H + ty;
  if (x<2*width && y<2*height) {
    int bx = (tx + 1)/2;
    int by = (ty + 1)/2;
    int bp = by*BW + bx;
    float wx = 0.25f + (tx&1)*0.50f;
    float wy = 0.25f + (ty&1)*0.50f;
    d_Result[y*newpitch + x] = wy*(wx*buffer[bp] + (1.0f-wx)*buffer[bp+1]) +
      (1.0f-wy)*(wx*buffer[bp+BW] + (1.0f-wx)*buffer[bp+BW+1]);
  }
}

__global__ void ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina;
  float scosa = scale*cosa;

  for (int y=ty;y<16;y+=8) {
    float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina;
    float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa;
    float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) -
      tex2D<float>(texObj, xpos-cosa, ypos-sina);
    float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) -
      tex2D<float>(texObj, xpos+sina, ypos-cosa);
    float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
    float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

    int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
    float horf = (tx - 1.5f)/4.0f - hori;
    float ihorf = 1.0f - horf;
    int veri = (y + 2)/4 - 1;
    float verf = (y - 1.5f)/4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi<7 ? angi+1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;

    int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated
    int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx>=2) {
      float grad1 = ihorf*grad;
      if (y>=2) {   // Upper left
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1, iangf*grad2);
	atomicAdd(buffer + p2,  angf*grad2);
      }
      if (y<=13) {  // Lower left
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+32, iangf*grad2);
	atomicAdd(buffer + p2+32,  angf*grad2);
      }
    }
    if (tx<=13) {
      float grad1 = horf*grad;
      if (y>=2) {    // Upper right
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1+8, iangf*grad2);
	atomicAdd(buffer + p2+8,  angf*grad2);
      }
      if (y<=13) {   // Lower right
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+40, iangf*grad2);
	atomicAdd(buffer + p2+40,  angf*grad2);
      }
    }
  }
  __syncthreads();

  // Normalize twice and suppress peaks first time
  float sum = buffer[idx]*buffer[idx];
  for (int i=1;i<=16;i*=2)
    sum += __shfl_xor(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
  tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

  sum = tsum1*tsum1;
  for (int i=1;i<=16;i*=2)
    sum += __shfl_xor(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();

  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
  float *desc = d_sift[bx].data;
  desc[idx] = tsum1 * rsqrtf(tsum2);
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}

__global__ void RescalePositions(SiftPoint *d_sift, int numPts, float scale)
{
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  if (num<numPts) {
    d_sift[num].xpos *= scale;
    d_sift[num].ypos *= scale;
    d_sift[num].scale *= scale;
  }
}


__global__ void ComputeOrientations(cudaTextureObject_t texObj, SiftPoint *d_Sift, int fstPts)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x + fstPts;
  float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
  if (tx<11)
    gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
  if (tx<64)
    hist[tx] = 0.0f;
  __syncthreads();
  float xp = d_Sift[bx].xpos - 5.0f;
  float yp = d_Sift[bx].ypos - 5.0f;
  int yd = tx/11;
  int xd = tx - yd*11;
  float xf = xp + xd;
  float yf = yp + yd;
  if (yd<11) {
    float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf);
    float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0);
    int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
    if (bin>31)
      bin = 0;
    float grad = sqrtf(dx*dx + dy*dy);
    atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
  }
  __syncthreads();
  int x1m = (tx>=1 ? tx-1 : tx+31);
  int x1p = (tx<=30 ? tx+1 : tx-31);
  if (tx<32) {
    int x2m = (tx>=2 ? tx-2 : tx+30);
    int x2p = (tx<=29 ? tx+2 : tx-30);
    hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
  }
  __syncthreads();
  if (tx<32) {
    float v = hist[32+tx];
    hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
  }
  __syncthreads();
  if (tx==0) {
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (int i=0;i<32;i++) {
      float v = hist[i];
      if (v>maxval1) {
	maxval2 = maxval1;
	maxval1 = v;
	i2 = i1;
	i1 = i;
      } else if (v>maxval2) {
	maxval2 = v;
	i2 = i;
      }
    }
    float val1 = hist[32+((i1+1)&31)];
    float val2 = hist[32+((i1+31)&31)];
    float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
    d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2>0.8f*maxval1 && false) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      if (idx<d_MaxNumPoints) {
	d_Sift[idx].xpos = d_Sift[bx].xpos;
	d_Sift[idx].ypos = d_Sift[bx].ypos;
	d_Sift[idx].scale = d_Sift[bx].scale;
	d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
	d_Sift[idx].subsampling = d_Sift[bx].subsampling;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////

__global__ void FindPointsMulti(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, int nScales, float subsampling, float lowestScale)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
  __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[96];

  int tx = threadIdx.x;
  int block = blockIdx.x/nScales;
  int scale = blockIdx.x - nScales*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  if (tx==0)
    cnt = 0;
  __syncthreads();

  int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
  for (int y=0;y<yloops;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr0 = ptr + max(0,ypos-1)*pitch;
    int yptr1 = ptr + ypos*pitch;
    int yptr2 = ptr + min(height-1,ypos+1)*pitch;
    {
      float d10 = d_Data0[yptr0];
      float d11 = d_Data0[yptr1];
      float d12 = d_Data0[yptr2];
      ymin1[tx] = fminf(fminf(d10, d11), d12);
      ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
    }
    {
      float d30 = d_Data0[yptr0 + 2*size];
      float d31 = d_Data0[yptr1 + 2*size];
      float d32 = d_Data0[yptr2 + 2*size];
      ymin3[tx] = fminf(fminf(d30, d31), d32);
      ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
    }
    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d21), d22)), ymin3[tx]);
    ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d21), d22)), ymax3[tx]);
    __syncthreads();
    if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
      if (d21<d_Threshold[1]) {
	float minv = fminf(fminf(fminf(ymin2[tx-1], ymin2[tx+1]), ymin1[tx]), ymin3[tx]);
	minv = fminf(fminf(minv, d20), d22);
	if (d21<minv) {
	  int pos = atomicInc(&cnt, 31);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
      if (d21>d_Threshold[0]) {
	float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx-1], ymax2[tx+1]), ymax1[tx]), ymax3[tx]);
	maxv = fmaxf(fmaxf(maxv, d20), d22);
	if (d21>maxv) {
	  int pos = atomicInc(&cnt, 31);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
    }
    __syncthreads();
  }
  if (tx<cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<d_EdgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]);
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]);
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = d_Scales[scale] * exp2f(pds*d_Factor);
      if (sc>=lowestScale) {
	unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
	d_Sift[idx].xpos = xpos + pdx;
	d_Sift[idx].ypos = ypos + pdy;
	d_Sift[idx].scale = sc;
	d_Sift[idx].sharpness = val + dval;
	d_Sift[idx].edgeness = edge;
	d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}


 __global__ void LaplaceMultiTex(cudaTextureObject_t texObj, float *d_Result, int width, int pitch, int height)
{
  __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_Kernel2 + scale*16;
  float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale;
  float x = xp-3.5;
  float y = yp+0.5;
  sdata1[tx] = kernel[4]*tex2D<float>(texObj, x, y) +
    kernel[3]*(tex2D<float>(texObj, x, y-1.0) + tex2D<float>(texObj, x, y+1.0)) +
    kernel[2]*(tex2D<float>(texObj, x, y-2.0) + tex2D<float>(texObj, x, y+2.0)) +
    kernel[1]*(tex2D<float>(texObj, x, y-3.0) + tex2D<float>(texObj, x, y+3.0)) +
    kernel[0]*(tex2D<float>(texObj, x, y-4.0) + tex2D<float>(texObj, x, y+4.0));
  __syncthreads();
  float *sdata2 = data2 + LAPLACE_W*scale;
  if (tx<LAPLACE_W) {
    sdata2[tx] = kernel[4]*sdata1[tx+4] +
      kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) +
      kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) +
      kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) +
      kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads();
  if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width)
    d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}


 __global__ void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_Kernel2 + scale*16;
  float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale;
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  int h = height-1;
  sdata1[tx] = kernel[4]*data[min(yp, h)*pitch] +
    kernel[3]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) +
    kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) +
    kernel[1]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) +
    kernel[0]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
  __syncthreads();
  float *sdata2 = data2 + LAPLACE_W*scale;
  if (tx<LAPLACE_W) {
    sdata2[tx] = kernel[4]*sdata1[tx+4] +
      kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) + kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) +
      kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) + kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads();
  if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width)
    d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}

 __global__ void LowPass(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float buffer[(LOWPASS_W + 2*LOWPASS_R)*LOWPASS_H];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  float *kernel = d_Kernel2;
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  float *buff = buffer + ty*(LOWPASS_W + 2*LOWPASS_R);
  int h = height-1;
  if (yp<height)
    buff[tx] = kernel[4]*data[min(yp, h)*pitch] +
      kernel[3]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) +
      kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) +
      kernel[1]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) +
      kernel[0]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
  __syncthreads();
  if (tx<LOWPASS_W && xp<width && yp<height) {
    d_Result[yp*pitch + xp] = kernel[4]*buff[tx+4] +
      kernel[3]*(buff[tx+3] + buff[tx+5]) + kernel[2]*(buff[tx+2] + buff[tx+6]) +
      kernel[1]*(buff[tx+1] + buff[tx+7]) + kernel[0]*(buff[tx+0] + buff[tx+8]);
  }
}
void InitCuda(int devNum)
{
//  int nDevices;
//  cudaGetDeviceCount(&nDevices);
//  if (!nDevices) {
//    std::cerr << "No CUDA devices available" << std::endl;
//    return;
//  }
//  devNum = std::min(nDevices-1, devNum);
//  deviceInit(devNum);
//  cudaDeviceProp prop;
//  cudaGetDeviceProperties(&prop, devNum);
//  printf("Device Number: %d\n", devNum);
//  printf("  Device name: %s\n", prop.name);
//  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
//  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
//  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
//	 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp)
{

  int totPts = 0;
  SafeCall(cudaMemcpyToSymbol(d_PointCounter, &totPts, sizeof(int)));
  SafeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

  const int nd = NUM_SCALES + 3;
  int w = img.width*(scaleUp ? 2 : 1);
  int h = img.height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;
  SafeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
  if (!scaleUp) {
    LowPass(lowImg, img, max(initBlur, 0.001f));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    SafeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    ScaleUp(upImg, img);
    LowPass(lowImg, upImg, max(initBlur, 0.001f));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    SafeCall(cudaMemcpyFromSymbol(&siftData.numPts, d_PointCounter, sizeof(int)));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
  }

  SafeCall(cudaFree(memoryTmp));

  if (siftData.h_data)
    SafeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost));
}

//extern double DynamicMain(CudaImage &img, SiftData &siftData, int numOctaves, double initBlur, float thresh, float lowestScale, float edgeLimit, float *memoryTmp);

void ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub)
{
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub);
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
  }
  if (lowestScale<subsampling*2.0f)
    ExtractSiftOctave(siftData, img, initBlur, thresh, lowestScale, subsampling, memoryTmp);
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nd = NUM_SCALES + 3;
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++)
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, baseBlur, diffScale, initBlur);
  int fstPts = 0;
  SafeCall(cudaMemcpyFromSymbol(&fstPts, d_PointCounter, sizeof(int)));
  double sigma = baseBlur*diffScale;
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling);

  int totPts = 0;
  SafeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>fstPts) {
    ComputeOrientations(texObj, siftData, fstPts, totPts);
    SafeCall(cudaMemcpyFromSymbol(&totPts, d_PointCounter, sizeof(int)));
    totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
    ExtractSiftDescriptors(texObj, siftData, fstPts, totPts, subsampling);
  }
  SafeCall(cudaDestroyTextureObject(texObj));
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint)*num;

  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    SafeCall(cudaMalloc((void **)&data.d_data, sz));
}

void FreeSiftData(SiftData &data)
{

  if (data.d_data!=NULL)
    SafeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);

  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
//#ifdef MANAGEDMEM
//  SiftPoint *h_data = data.m_data;
//#else
//  SiftPoint *h_data = data.h_data;
//  if (data.h_data==NULL) {
//    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
//    SafeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost));
//    data.h_data = h_data;
//  }
//#endif
//  for (int i=0;i<data.numPts;i++) {
//    printf("xpos         = %.2f\n", h_data[i].xpos);
//    printf("ypos         = %.2f\n", h_data[i].ypos);
//    printf("scale        = %.2f\n", h_data[i].scale);
//    printf("sharpness    = %.2f\n", h_data[i].sharpness);
//    printf("edgeness     = %.2f\n", h_data[i].edgeness);
//    printf("orientation  = %.2f\n", h_data[i].orientation);
//    printf("score        = %.2f\n", h_data[i].score);
//    float *siftData = (float*)&h_data[i].data;
//    for (int j=0;j<8;j++) {
//      if (j==0)
//	printf("data = ");
//      else
//	printf("       ");
//      for (int k=0;k<16;k++)
//	if (siftData[j+8*k]<0.05)
//	  printf(" .   ");
//	else
//	  printf("%.2f ", siftData[j+8*k]);
//      printf("\n");
//    }
//  }
//  printf("Number of available points: %d\n", data.numPts);
//  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;
  SafeCall(cudaMemcpyToSymbol(d_Kernel1, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);

  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W, SCALEUP_H);
  ScaleUp<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);

  return 0.0;
}


double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts)
{
  dim3 blocks(totPts - fstPts);
  dim3 threads(128);

  ComputeOrientations<<<blocks, threads>>>(texObj, siftData.d_data, fstPts);

  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling)
{
  dim3 blocks(totPts - fstPts);
  dim3 threads(16, 8);

  ExtractSiftDescriptors<<<blocks, threads>>>(texObj, siftData.d_data, fstPts, subsampling);

  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositions<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);

  return 0.0;
}

double LowPass(CudaImage &res, CudaImage &src, float scale)
{
  float kernel[16];
  float kernelSum = 0.0f;
  float ivar2 = 1.0f/(2.0f*scale*scale);
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
    kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
    kernelSum += kernel[j+LOWPASS_R];
  }
  for (int j=-LOWPASS_R;j<=LOWPASS_R;j++)
    kernel[j+LOWPASS_R] /= kernelSum;
  SafeCall(cudaMemcpyToSymbol(d_Kernel2, kernel, 12*16*sizeof(float)));
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);

  return 0.0;
}

//==================== Multi-scale functions ===================//

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, float baseBlur, float diffScale, float initBlur)
{
  float kernel[12*16];
  float scale = baseBlur;
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++) {
      kernel[16*i+j+LAPLACE_R] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += kernel[16*i+j+LAPLACE_R];
    }
    for (int j=-LAPLACE_R;j<=LAPLACE_R;j++)
      kernel[16*i+j+LAPLACE_R] /= kernelSum;
    scale *= diffScale;
  }
  SafeCall(cudaMemcpyToSymbol(d_Kernel2, kernel, 12*16*sizeof(float)));
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);

  LaplaceMultiMem<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height);

  return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
  float threshs[2] = { thresh, -thresh };
  float scales[NUM_SCALES];
  float diffScale = pow(2.0f, factor);
  for (int i=0;i<NUM_SCALES;i++) {
    scales[i] = scale;
    scale *= diffScale;
  }
  SafeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));
  SafeCall(cudaMemcpyToSymbol(d_EdgeLimit, &edgeLimit, sizeof(float)));
  SafeCall(cudaMemcpyToSymbol(d_Scales, scales, sizeof(float)*NUM_SCALES));
  SafeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2);

  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, NUM_SCALES, subsampling, lowestScale);

  return 0.0;
}
__global__ void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x;
  const int p2 = blockIdx.y*16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = 16*ty + tx;
  if (ty<8)
    siftPoint[i] = ptr1[i];
  __syncthreads();
  float sum = 0.0f;
  if (p2<numPts2)
    for (int j=0;j<8;j++)
      sum += siftPoint[16*j+tx] * ptr2[16*j+tx];
  sums[i] = sum;
  __syncthreads();
  if (tx<8)
    sums[i] += sums[i+8];
  __syncthreads();
  if (tx<4)
    sums[i] += sums[i+4];
  __syncthreads();
  if (ty==0) {
    sum = sums[16*tx+0] + sums[16*tx+1] + sums[16*tx+2] + sums[16*tx+3];
    corrData[p1*gridDim.y*16 + blockIdx.y*16 + tx] = sum;
  }
  __syncthreads();
}


__global__ void MatchSiftPoints2(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoints1[16*128];
  __shared__ float siftPoints2[16*128];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const float *ptr1 = sift1[min(numPts1-1,blockIdx.x*16 + ty)].data;
  const float *ptr2 = sift2[min(numPts2-1,blockIdx.y*16 + ty)].data;
  for (int i=0;i<8;i++) {
    siftPoints1[128*ty+16*i+tx] = ptr1[16*i+tx];
    siftPoints2[128*ty+16*i+tx] = ptr2[16*i+tx];
  }
  __syncthreads();
  const int p1 = blockIdx.x*16 + ty;
  const int p2 = blockIdx.y*16 + tx;
  const float *pt1 = &siftPoints1[ty*128];
  const float *pt2 = &siftPoints2[tx*128];
  float sum = 0.0f;
  for (int i=0;i<128;i++) {
    int itx = (i + tx)&127; // avoid bank conflicts
    sum += pt1[itx]*pt2[itx];
  }
  if (p1<numPts1)
    corrData[p1*gridDim.y*16 + p2] = (p2<numPts2 ? sum : -1.0f);
}

__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize)
{
  __shared__ float maxScore[16*16];
  __shared__ float maxScor2[16*16];
  __shared__ int maxIndex[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*16 + tx;
  int p1 = blockIdx.x*16 + threadIdx.y;
  p1 = (p1>=numPts1 ? numPts1-1 : p1);
  maxScore[idx] = -1.0f;
  maxScor2[idx] = -1.0f;
  maxIndex[idx] = -1;
  __syncthreads();
  float *corrs = &corrData[p1*corrWidth];
  for (int i=tx;i<corrWidth;i+=16) {
    float val = corrs[i];
    if (val>maxScore[idx]) {
      maxScor2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    } else if (val>maxScor2[idx])
      maxScor2[idx] = val;
  }
  //if (p1==1)
  //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n",
  //	   tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  __syncthreads();
  for (int len=8;len>0;len/=2) {
    if (tx<8) {
      float val = maxScore[idx+len];
      int i = maxIndex[idx+len];
      if (val>maxScore[idx]) {
	maxScor2[idx] = maxScore[idx];
	maxScore[idx] = val;
	maxIndex[idx] = i;
      } else if (val>maxScor2[idx])
	maxScor2[idx] = val;
      float va2 = maxScor2[idx+len];
      if (va2>maxScor2[idx])
	maxScor2[idx] = va2;
    }
    __syncthreads();
    //if (p1==1 && tx<len)
    //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n",
    //	     tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  }
  if (tx==6)
    sift1[p1].score = maxScore[ty*16];
  if (tx==7)
    sift1[p1].ambiguity = maxScor2[ty*16] / (maxScore[ty*16] + 1e-6);
  if (tx==8)
    sift1[p1].match = maxIndex[ty*16];
  if (tx==9)
    sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
  if (tx==10)
    sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  __syncthreads();
  //if (tx==0)
  //  printf("index = %d/%d, score = %.2f, ambiguity = %.2f, match = %d\n",
  //	p1, numPts1, sift1[p1].score, sift1[p1].ambiguity, sift1[p1].match);
}

template <int size>
__device__ void InvertMatrix(float elem[size][size], float res[size][size])
{
  int indx[size];
  float b[size];
  float vv[size];
  for (int i=0;i<size;i++)
    indx[i] = 0;
  int imax = 0;
  float d = 1.0;
  for (int i=0;i<size;i++) { // find biggest element for each row
    float big = 0.0;
    for (int j=0;j<size;j++) {
      float temp = fabs(elem[i][j]);
      if (temp>big)
	big = temp;
    }
    if (big>0.0)
      vv[i] = 1.0/big;
    else
      vv[i] = 1e16;
  }
  for (int j=0;j<size;j++) {
    for (int i=0;i<j;i++) { // i<j
      float sum = elem[i][j]; // i<j (lower left)
      for (int k=0;k<i;k++) // k<i<j
	sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i<j (lower left)
    }
    float big = 0.0;
    for (int i=j;i<size;i++) { // i>=j
      float sum = elem[i][j]; // i>=j (upper right)
      for (int k=0;k<j;k++) // k<j<=i
	sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i>=j (upper right)
      float dum = vv[i]*fabs(sum);
      if (dum>=big) {
	big = dum;
	imax = i;
      }
    }
    if (j!=imax) { // imax>j
      for (int k=0;k<size;k++) {
	float dum = elem[imax][k]; // upper right and lower left
	elem[imax][k] = elem[j][k];
	elem[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (elem[j][j]==0.0)  // j==j (upper right)
      elem[j][j] = 1e-16;
    if (j!=(size-1)) {
      float dum = 1.0/elem[j][j];
      for (int i=j+1;i<size;i++) // i>j
	elem[i][j] *= dum; // i>j (upper right)
    }
  }
  for (int j=0;j<size;j++) {
    for (int k=0;k<size;k++)
      b[k] = 0.0;
    b[j] = 1.0;
    int ii = -1;
    for (int i=0;i<size;i++) {
      int ip = indx[i];
      float sum = b[ip];
      b[ip] = b[i];
      if (ii!=-1)
	for (int j=ii;j<i;j++)
	  sum -= elem[i][j]*b[j]; // i>j (upper right)
      else if (sum!=0.0)
        ii = i;
      b[i] = sum;
    }
    for (int i=size-1;i>=0;i--) {
      float sum = b[i];
      for (int j=i+1;j<size;j++)
	sum -= elem[i][j]*b[j]; // i<j (lower left)
      b[i] = sum/elem[i][i]; // i==i (upper right)
    }
    for (int i=0;i<size;i++)
      res[i][j] = b[i];
  }
}

__global__ void ComputeHomographies(float *coord, int *randPts, float *homo,
  int numPts)
{
  float a[8][8], ia[8][8];
  float b[8];
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int idx = blockDim.x*bx + tx;
  const int numLoops = blockDim.x*gridDim.x;
  for (int i=0;i<4;i++) {
    int pt = randPts[i*numLoops+idx];
    float x1 = coord[pt+0*numPts];
    float y1 = coord[pt+1*numPts];
    float x2 = coord[pt+2*numPts];
    float y2 = coord[pt+3*numPts];
    float *row1 = a[2*i+0];
    row1[0] = x1;
    row1[1] = y1;
    row1[2] = 1.0;
    row1[3] = row1[4] = row1[5] = 0.0;
    row1[6] = -x2*x1;
    row1[7] = -x2*y1;
    float *row2 = a[2*i+1];
    row2[0] = row2[1] = row2[2] = 0.0;
    row2[3] = x1;
    row2[4] = y1;
    row2[5] = 1.0;
    row2[6] = -y2*x1;
    row2[7] = -y2*y1;
    b[2*i+0] = x2;
    b[2*i+1] = y2;
  }
  InvertMatrix<8>(a, ia);
  __syncthreads();
  for (int j=0;j<8;j++) {
    float sum = 0.0f;
    for (int i=0;i<8;i++)
      sum += ia[j][i]*b[i];
    homo[j*numLoops+idx] = sum;
  }
  __syncthreads();
}

#define TESTHOMO_TESTS 16 // number of tests per block,  alt. 32, 32
#define TESTHOMO_LOOPS 16 // number of loops per block,  alt.  8, 16

__global__ void TestHomographies(float *d_coord, float *d_homo,
  int *d_counts, int numPts, float thresh2)
{
  __shared__ float homo[8*TESTHOMO_LOOPS];
  __shared__ int cnts[TESTHOMO_TESTS*TESTHOMO_LOOPS];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = blockIdx.y*blockDim.y + tx;
  const int numLoops = blockDim.y*gridDim.y;
  if (ty<8 && tx<TESTHOMO_LOOPS)
    homo[tx*8+ty] = d_homo[idx+ty*numLoops];
  __syncthreads();
  float a[8];
  for (int i=0;i<8;i++)
    a[i] = homo[ty*8+i];
  int cnt = 0;
  for (int i=tx;i<numPts;i+=TESTHOMO_TESTS) {
    float x1 = d_coord[i+0*numPts];
    float y1 = d_coord[i+1*numPts];
    float x2 = d_coord[i+2*numPts];
    float y2 = d_coord[i+3*numPts];
    float nomx = __fmul_rz(a[0],x1) + __fmul_rz(a[1],y1) + a[2];
    float nomy = __fmul_rz(a[3],x1) + __fmul_rz(a[4],y1) + a[5];
    float deno = __fmul_rz(a[6],x1) + __fmul_rz(a[7],y1) + 1.0f;
    float errx = __fmul_rz(x2,deno) - nomx;
    float erry = __fmul_rz(y2,deno) - nomy;
    float err2 = __fmul_rz(errx,errx) + __fmul_rz(erry,erry);
    if (err2<__fmul_rz(thresh2,__fmul_rz(deno,deno)))
      cnt ++;
  }
  int kty = TESTHOMO_TESTS*ty;
  cnts[kty + tx] = cnt;
  __syncthreads();
  int len = TESTHOMO_TESTS/2;
  while (len>0) {
    if (tx<len)
      cnts[kty + tx] += cnts[kty + tx + len];
    len /= 2;
    __syncthreads();
  }
  if (tx<TESTHOMO_LOOPS && ty==0)
    d_counts[idx] = cnts[TESTHOMO_TESTS*tx];
  __syncthreads();
}

//================= Host matching functions =====================//

double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
#ifdef MANAGEDMEM
  SiftPoint *d_sift = data.m_data;
#else
  if (data.d_data==NULL)
    return 0.0f;
  SiftPoint *d_sift = data.d_data;
#endif

  numLoops = iDivUp(numLoops,16)*16;
  int numPts = data.numPts;
  if (numPts<8)
    return 0.0f;
  int numPtsUp = iDivUp(numPts, 16)*16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4*sizeof(int)*numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
  SafeCall(cudaMalloc((void **)&d_coord, 4*sizeof(float)*numPtsUp));
  SafeCall(cudaMalloc((void **)&d_randPts, randSize));
  SafeCall(cudaMalloc((void **)&d_homo, 8*sizeof(float)*numLoops));
  h_randPts = (int*)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float)*numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float)*numPtsUp);
  SafeCall(cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
  SafeCall(cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
  int *validPts = (int *)malloc(sizeof(int)*numPts);
  int numValid = 0;
  for (int i=0;i<numPts;i++) {
    if (h_scores[i]>minScore && h_ambiguities[i]<maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid>=8) {
    for (int i=0;i<numLoops;i++) {
      int p1 = rand() % numValid;
      int p2 = rand() % numValid;
      int p3 = rand() % numValid;
      int p4 = rand() % numValid;
      while (p2==p1) p2 = rand() % numValid;
      while (p3==p1 || p3==p2) p3 = rand() % numValid;
      while (p4==p1 || p4==p2 || p4==p3) p4 = rand() % numValid;
      h_randPts[i+0*numLoops] = validPts[p1];
      h_randPts[i+1*numLoops] = validPts[p2];
      h_randPts[i+2*numLoops] = validPts[p3];
      h_randPts[i+3*numLoops] = validPts[p4];
    }
    SafeCall(cudaMemcpy(d_randPts, h_randPts, randSize, cudaMemcpyHostToDevice));
    SafeCall(cudaMemcpy2D(&d_coord[0*numPtsUp], szFl, &d_sift[0].xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    SafeCall(cudaMemcpy2D(&d_coord[1*numPtsUp], szFl, &d_sift[0].ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    SafeCall(cudaMemcpy2D(&d_coord[2*numPtsUp], szFl, &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    SafeCall(cudaMemcpy2D(&d_coord[3*numPtsUp], szFl, &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    ComputeHomographies<<<numLoops/16, 16>>>(d_coord, d_randPts, d_homo, numPtsUp);
    SafeCall(cudaThreadSynchronize());

    dim3 blocks(1, numLoops/TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, d_randPts, numPtsUp, thresh*thresh);
    SafeCall(cudaThreadSynchronize());

    SafeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, cudaMemcpyDeviceToHost));
    int maxIndex = -1, maxCount = -1;
    for (int i=0;i<numLoops;i++)
      if (h_randPts[i]>maxCount) {
	maxCount = h_randPts[i];
	maxIndex = i;
      }
    *numMatches = maxCount;
    SafeCall(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], sizeof(float)*numLoops, szFl, 8, cudaMemcpyDeviceToHost));
  }
  free(validPts);
  free(h_randPts);
  SafeCall(cudaFree(d_homo));
  SafeCall(cudaFree(d_randPts));
  SafeCall(cudaFree(d_coord));

#ifdef VERBOSE
  printf("FindHomography time =         %.2f ms\n", gpuTime);
#endif

}

double MatchSiftData(SiftData &data1, SiftData &data2)
{

  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2)
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif

  float *d_corrData;
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  SafeCall(cudaMalloc((void **)&d_corrData, corrSize));

  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);

  SafeCall(cudaThreadSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  SafeCall(cudaThreadSynchronize());

  SafeCall(cudaFree(d_corrData));
  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    SafeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }
}

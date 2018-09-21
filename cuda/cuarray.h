#ifndef DEVICE_ARRAY_CUH__
#define DEVICE_ARRAY_CUH__

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#if defined(__GNUC__)
    #define SafeCall(expr)  ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
    #define SafeCall(expr)  ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "") {
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

template<class T> struct PtrSz {

	__device__ T& operator[](int x);
	__device__ operator T*();
	__device__ operator const T*() const;
	__device__ T operator[](int x) const;

	T* data;
	size_t size;
};

template<class T> class DeviceArray {
public:

	DeviceArray();
	~DeviceArray();
	DeviceArray(size_t size, bool managed = false);

	void create(size_t size, bool managed = false);
	void upload(void* host_ptr, size_t size);
	void download(void* host_ptr);
	void download(void* host_ptr, size_t size);
	void zero();
	T* last();
	void Release();
	void copyTo(DeviceArray<T>& other) const;
	bool empty() const;
	size_t size() const;
	T operator[](int idx) const ;
	DeviceArray<T>& operator=(const DeviceArray<T>& other);
	operator T*();
	operator const T*() const;
	operator PtrSz<T>();
	operator PtrSz<T>() const;

private:

	void* data;
	size_t mSize;
	std::atomic<int>* ref;
};

template<class T> struct PtrStep {

	__device__ T* ptr(int y = 0);
	__device__ const T* ptr(int y = 0) const;

	T* data;
	size_t step;
};

template<class T> struct PtrStepSz {

	__device__ T* ptr(int y = 0);
	__device__ const T* ptr(int y = 0) const;

	T* data;
	int cols;
	int rows;
	size_t step;
};

template<class T>
class DeviceArray2D {
public:
	DeviceArray2D();
	~DeviceArray2D();
	DeviceArray2D(int cols, int rows);

	void create(int cols, int rows);
	void upload(void* host_ptr, size_t host_step, int cols, int rows);
	void download(void* host_ptr, size_t host_step) const;
	void zero();
	void release();
	void swap(DeviceArray2D<T> & other);
	void copyTo(DeviceArray2D<T>& other) const;
	DeviceArray2D<T>& operator=(const DeviceArray2D<T>& other);
	bool empty() const;
	size_t step() const;
	int cols() const;
	int rows() const;
	void* data() const;

	operator T*();
	operator const T*() const;
	operator PtrStep<T>();
	operator PtrStep<T>() const;
	operator PtrStepSz<T>();
	operator PtrStepSz<T>() const;

private:

	void* mpData;
	size_t mStep;
	int mCols, mRows;
	std::atomic<int>* mpRef;
};

//------------------------------------------------------------------
// PtrSz
//------------------------------------------------------------------
template<class T> __device__ inline T& PtrSz<T>::operator [](int x) {
	return data[x];
}

template<class T> __device__ inline T PtrSz<T>::operator [](int x) const {
	return data[x];
}

template<class T> __device__ inline PtrSz<T>::operator T*() {
	return data;
}

template<class T> __device__ inline PtrSz<T>::operator const T*() const {
	return data;
}

//------------------------------------------------------------------
// DeviceArray
//------------------------------------------------------------------
template<class T> inline DeviceArray<T>::DeviceArray() :
		data(nullptr), ref(nullptr), mSize(0) {
}

template<class T> inline DeviceArray<T>::DeviceArray(size_t size, bool managed) :
		data(nullptr), ref(nullptr), mSize(size) {
	if(managed)
		create(size, true);
	else
		create(size);
}

template<class T> inline DeviceArray<T>::~DeviceArray() {
	Release();
}

template<class T> inline void DeviceArray<T>::create(size_t size, bool managed) {
	if (!empty())
		Release();
	mSize = size;
	if(managed)
		SafeCall(cudaMallocManaged((void**) &data, sizeof(T) * mSize));
	else
		SafeCall(cudaMalloc(&data, sizeof(T) * mSize));
	ref = new std::atomic<int>(1);
}

template<class T> inline void DeviceArray<T>::upload(void* host_ptr, size_t size) {
	if (size > mSize)
		return;
	SafeCall(cudaMemcpy(data, host_ptr, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template<class T> inline void DeviceArray<T>::download(void* host_ptr) {
	SafeCall(cudaMemcpy(host_ptr, data, sizeof(T) * mSize,	cudaMemcpyDeviceToHost));
}

template<class T> inline void DeviceArray<T>::download(void* host_ptr, size_t size) {
	SafeCall(cudaMemcpy(host_ptr, data, sizeof(T) * size,	cudaMemcpyDeviceToHost));
}

template<class T> inline void DeviceArray<T>::zero() {
	SafeCall(cudaMemset(data, 0, sizeof(T) * mSize));
}

template<class T> inline T* DeviceArray<T>::last() {
	return &((T*)data)[mSize - 1];
}


template<class T> inline void DeviceArray<T>::Release() {
	if (ref && --*ref == 0) {
		delete ref;
		if (!empty()) {
			SafeCall(cudaFree(data));
		}
	}
	mSize = 0;
	data = ref = 0;
}

template<class T> inline void DeviceArray<T>::copyTo(DeviceArray<T>& other) const {
	if (empty()) {
		other.Release();
		return;
	}
	other.create(mSize);
	SafeCall(cudaMemcpy(other.data, data, sizeof(T) * mSize, cudaMemcpyDeviceToDevice));
}

template<class T> inline bool DeviceArray<T>::empty() const {
	return !data;
}

template<class T> inline size_t DeviceArray<T>::size() const {
	return mSize;
}

template<class T> inline T DeviceArray<T>::operator[](int idx) const {
	return ((T*)data)[idx];
}

template<class T> inline DeviceArray<T>& DeviceArray<T>::operator=(const DeviceArray<T>& other) {
	if(this != &other) {
		if(other.ref)
			++*other.ref;
		Release();
		ref = other.ref;
		mSize = other.mSize;
		data = other.data;
	} return *this;
}

template<class T> inline DeviceArray<T>::operator T*() {
	return (T*)data;
}

template<class T> inline DeviceArray<T>::operator const T*() const {
	return (T*)data;
}

template<class T> inline DeviceArray<T>::operator PtrSz<T>() {
	PtrSz<T> ps;
	ps.data = (T*) data;
	ps.size = mSize;
	return ps;
}

template<class T> inline DeviceArray<T>::operator PtrSz<T>() const {
	PtrSz<T> ps;
	ps.data = (T*) data;
	ps.size = mSize;
	return ps;
}

//------------------------------------------------------------------
// PtrStep
//------------------------------------------------------------------
template<class T> __device__ inline T* PtrStep<T>::ptr(int y) {
	return (T*) ((char*) data + y * step);
}

template<class T> __device__ inline const T* PtrStep<T>::ptr(int y) const {
	return (T*) ((char*) data + y * step);
}

//------------------------------------------------------------------
// PtrStepSz
//------------------------------------------------------------------
template<class T> __device__ inline T* PtrStepSz<T>::ptr(int y) {
	return (T*) ((char*) data + y * step);
}

template<class T> __device__ inline const T* PtrStepSz<T>::ptr(int y) const {
	return (T*) ((char*) data + y * step);
}

//------------------------------------------------------------------
// DeviceArray2D
//------------------------------------------------------------------
template<class T> inline DeviceArray2D<T>::DeviceArray2D():
		mpData(nullptr), mpRef(nullptr), mStep(0), mCols(0), mRows(0) {
}

template<class T> inline DeviceArray2D<T>::DeviceArray2D(int cols, int rows):
		mpData(nullptr), mpRef(nullptr), mStep(0), mCols(cols), mRows(rows) {
	create(mCols, mRows);
}

template<class T> inline DeviceArray2D<T>::~DeviceArray2D() {
	release();
}

template<class T> inline void DeviceArray2D<T>::create(int cols, int rows) {
	if(cols > 0 && rows > 0) {
		if(!empty())
			release();
		mCols = cols;
		mRows = rows;
		SafeCall(cudaMallocPitch(&mpData, &mStep, sizeof(T) * mCols, mRows));
		mpRef = new std::atomic<int>(1);
	}
}

template<class T> inline void DeviceArray2D<T>::upload(void* host_ptr, size_t host_step, int cols, int rows) {
	if(empty())
		create(cols, rows);
	SafeCall(cudaMemcpy2D(mpData, mStep, host_ptr, host_step, sizeof(T) * mCols, mRows, cudaMemcpyHostToDevice));
}

template<class T> inline void DeviceArray2D<T>::swap(DeviceArray2D<T> & other) {
	std::swap(mpData, other.mpData);
	std::swap(mCols, other.mCols);
	std::swap(mRows, other.mRows);
	std::swap(mStep, other.mStep);
	std::swap(mpRef, other.mpRef);
}

template<class T> inline void DeviceArray2D<T>::zero() {
	SafeCall(cudaMemset2D(mpData, mStep, 0, sizeof(T) * mCols, mRows));
}

template<class T> inline void DeviceArray2D<T>::download(void* host_ptr, size_t host_step) const {
	if(empty())
		return;
	SafeCall(cudaMemcpy2D(host_ptr, host_step, mpData, mStep, sizeof(T) * mCols, mRows, cudaMemcpyDeviceToHost));
}

template<class T> inline void DeviceArray2D<T>::release() {
	if(mpRef && --*mpRef == 0) {
		delete mpRef;
		if(!empty())
			SafeCall(cudaFree(mpData));
	}
	mCols = mRows = mStep = 0;
	mpData = mpRef = 0;
}

template<class T> inline void DeviceArray2D<T>::copyTo(DeviceArray2D<T>& other) const {
	if(empty())
		other.release();
	other.create(mCols, mRows);
	SafeCall(cudaMemcpy2D(other.mpData, other.mStep, mpData, mStep, sizeof(T) * mCols, mRows, cudaMemcpyDeviceToDevice));
}

template<class T> inline DeviceArray2D<T>& DeviceArray2D<T>::operator=(const DeviceArray2D<T>& other) {
	if(this != &other) {
		if(other.mpRef)
			++*other.mpRef;
		release();
		mpData = other.mpData;
		mStep = other.mStep;
		mCols = other.mCols;
		mRows = other.mRows;
		mpRef = other.mpRef;
	} return *this;
}

template<class T> inline bool DeviceArray2D<T>::empty() const {
	return !mpData;
};

template<class T> inline size_t DeviceArray2D<T>::step() const {
	return mStep;
};

template<class T> inline int DeviceArray2D<T>::cols() const {
	return mCols;
};

template<class T> inline int DeviceArray2D<T>::rows() const {
	return mRows;
};

template<class T> inline void* DeviceArray2D<T>::data() const {
	return (void*) mpData;
}

template<class T> inline DeviceArray2D<T>::operator T*() {
	return (T*)mpData;
}

template<class T> inline DeviceArray2D<T>::operator const T*() const {
	return (T*)mpData;
}

template<class T> inline DeviceArray2D<T>::operator PtrStep<T>() {
	PtrStep<T> ps;
	ps.data = (T*)mpData;
	ps.step = mStep;
	return ps;
}

template<class T> inline DeviceArray2D<T>::operator PtrStep<T>() const {
	PtrStep<T> ps;
	ps.data = (T*)mpData;
	ps.step = mStep;
	return ps;
}

template<class T> inline DeviceArray2D<T>::operator PtrStepSz<T>() {
	PtrStepSz<T> psz;
	psz.data = (T*)mpData;
	psz.cols = mCols;
	psz.rows = mRows;
	psz.step = mStep;
	return psz;
}

template<class T> inline DeviceArray2D<T>::operator PtrStepSz<T>() const {
	PtrStepSz<T> psz;
	psz.data = (T*)mpData;
	psz.cols = mCols;
	psz.rows = mRows;
	psz.step = mStep;
	return psz;
}

#endif /* DEVICE_ARRAY_H_ */

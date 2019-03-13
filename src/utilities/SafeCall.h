#ifndef SAFECALL_H__
#define SAFECALL_H__

#include <iostream>
#include <cuda_runtime.h>

#if defined(__GNUC__)
    #define safe_call(expr)  ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
    #define safe_call(expr)  ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "") {
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

template<class T, class U>
constexpr int div_up(T a, U b) {
	return (int)((a + b - 1) / b);
}

#endif

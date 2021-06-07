#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor.")

#ifndef TORCH_CUSOLVER_CHECK
#define TORCH_CUSOLVER_CHECK(EXPR)                              \
  do {                                                          \
    cusolverStatus_t __err = EXPR;                              \
    TORCH_CHECK(__err == CUSOLVER_STATUS_SUCCESS,               \
                "CUDA error: ",                                 \
                cusolverGetErrorString(__err),                  \
                " when calling `" #EXPR "`");                   \
  } while (0)

static char* cusolverGetErrorString(cusolverStatus_t error) {
  if (error == CUSOLVER_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUSOLVER_STATUS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  }
  if (error == CUSOLVER_STATUS_ALLOC_FAILED) {
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  }
  if (error == CUSOLVER_STATUS_INVALID_VALUE) {
    return "CUSOLVER_STATUS_INVALID_VALUE";
  }
  if (error == CUSOLVER_STATUS_ARCH_MISMATCH) {
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  }
  if (error == CUSOLVER_STATUS_EXECUTION_FAILED) {
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  }
  if (error == CUSOLVER_STATUS_INTERNAL_ERROR) {
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  }
  if (error == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }
  return "<unknown>";
}
#endif

//#ifndef TORCH_CUDABLAS_CHECK
#define TORCH_CUDABLAS_CHECK(EXPR)                                      \
  do {                                                          \
    cublasStatus_t __err = EXPR;                              \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CuBLAS error: ",                               \
                cublasGetErrorString(__err),                    \
                " when calling `" #EXPR "`");                   \
  } while (0)


static char const * cublasGetErrorString(cublasStatus_t error) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}
//#endif


inline bool is_fortran_contig(const torch::Tensor &matrix) {
    return matrix.stride(0) == 1;
}

inline __device__ int2 tri_index_lower(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        linear_index - row * (row + 1) / 2,
        row
    );
}

inline __device__ int2 tri_index_upper(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        row,
        linear_index - row * (row + 1) / 2
    );
}

inline int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}





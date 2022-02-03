#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>


void cublas_2d_copy_to_dev_async (
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& host_tensor,
    const int lda, torch::Tensor& dev_tensor,
    const int ldb,
    const at::cuda::CUDAStream &stream);

void cublas_2d_copy_to_dev (
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& host_tensor,
    const int lda, torch::Tensor& dev_tensor,
    const int ldb);

void cublas_2d_copy_to_host_async(
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& dev_tensor,
    const int lda, torch::Tensor& host_tensor,
    const int ldb,
    const at::cuda::CUDAStream &stream);

void cublas_2d_copy_to_host(
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& dev_tensor,
    const int lda, torch::Tensor& host_tensor,
    const int ldb);

template<typename scalar_t>
inline void trsm(cublasHandle_t cublas_handle,
                 cublasSideMode_t side,
                 cublasFillMode_t uplo,
                 cublasOperation_t trans,
                 cublasDiagType_t diag,
                 int m,
                 int n,
                 const scalar_t *alpha,
                 const scalar_t *A,
                 int lda,
                 scalar_t *B,
                 int ldb);

void cublas_trsm(const Tensor& A, const Tensor& B, torch::Scalar alpha, bool left, bool upper, bool transpose, bool unitriangular, int m, int n, int lda, int ldb);

template<typename scalar_t>
inline void trmm(cublasHandle_t cublas_handle,
                 cublasSideMode_t side,
                 cublasFillMode_t uplo,
                 cublasOperation_t trans,
                 cublasDiagType_t diag,
                 int m,
                 int n,
                 const scalar_t *alpha,
                 const scalar_t *A,
                 int lda,
                 scalar_t *B,
                 int ldb,
                 scalar_t *C,
                 int ldc);

void cublas_trmm(const Tensor& A, const Tensor& B, const Tensor& C, bool left, bool upper, bool transpose, bool unitriangular, int m, int n, int lda, int ldb, int ldc);

inline void gemm(cublasHandle_t cublas_handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const scalar_t *alpha,
                 const scalar_t *A,
                 int lda,
                 const scalar_t *B,
                 int ldb,
                 const scalar_t *beta,
                 scalar_t *C,
                 int ldc);

void cublas_gemm(const Tensor& A, int lda, bool transa, const Tensor& B, int ldb, bool transb, const Tensor& C, int ldc, int m, int n, int k, Scalar alpha, Scalar beta);

template<typename scalar_t>
inline void syrk(cublasHandle_t cublas_handle,
                 cublasFillMode_t uplo,
                 cublasOperation_t trans,
                 int n,
                 int k,
                 const scalar_t *alpha,
                 const scalar_t *A,
                 int lda,
                 const scalar_t *beta,
                 scalar_t *C,
                 int ldc);

void cublas_syrk(const Tensor& A, int lda, const Tensor& C, int ldc, Scalar alpha, Scalar beta, bool upper, bool transpose, int n, int k);

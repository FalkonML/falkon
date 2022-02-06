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
void trsm(cublasHandle_t cublas_handle,
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

void cublas_trsm(const torch::Tensor& A, const torch::Tensor& B, torch::Scalar alpha, bool left, bool upper, bool transpose, bool unitriangular, int m, int n, int lda, int ldb);

template<typename scalar_t>
void trmm(cublasHandle_t cublas_handle,
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

void cublas_trmm(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, bool left, bool upper, bool transpose, bool unitriangular, torch::Scalar alpha, int m, int n, int lda, int ldb, int ldc);

template<typename scalar_t>
void gemm(cublasHandle_t cublas_handle,
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

void cublas_gemm(const torch::Tensor& A, int lda, bool transa, const torch::Tensor& B, int ldb, bool transb, const torch::Tensor& C, int ldc, int m, int n, int k, torch::Scalar alpha, torch::Scalar beta);

template<typename scalar_t>
void syrk(cublasHandle_t cublas_handle,
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

void cublas_syrk(const torch::Tensor& A, int lda, const torch::Tensor& C, int ldc, torch::Scalar alpha, torch::Scalar beta, bool upper, bool transpose, int n, int k);

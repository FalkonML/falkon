#pragma once

#include <cublas_v2.h>

namespace falkon {
namespace ops {

template<typename scalar_t>
void trsm(
    cublasHandle_t cublas_handle,
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

template<typename scalar_t>
void trmm(
    cublasHandle_t cublas_handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m,
    int n,
    const scalar_t *alpha,
    const scalar_t *A,
    int lda,
    const scalar_t *B,
    int ldb,
    scalar_t *C,
    int ldc);

template<typename scalar_t>
void gemm(
    cublasHandle_t cublas_handle,
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

template<typename scalar_t>
void syrk(
    cublasHandle_t cublas_handle,
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

} // namespace ops
} // namespace falkon

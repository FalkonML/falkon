#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {
void cublas_2d_copy_to_dev_async (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb);
void cublas_2d_copy_to_dev (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb);
void cublas_2d_copy_to_host_async(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb);
void cublas_2d_copy_to_host(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb);
void cublas_trsm(
        const at::Tensor& A,
        at::Tensor& B,
        const at::Scalar& alpha,
        bool left,
        bool upper,
        bool transpose,
        bool unitriangular,
        int64_t m,
        int64_t n,
        int64_t lda,
        int64_t ldb);
void cublas_trmm(
        const at::Tensor& A,
        const at::Tensor& B,
        at::Tensor& C,
        bool left,
        bool upper,
        bool transpose,
        bool unitriangular,
        const at::Scalar& alpha,
        int64_t m,
        int64_t n,
        int64_t lda,
        int64_t ldb,
        int64_t ldc);
void cublas_gemm(
        const at::Tensor& A,
        int64_t lda,
        bool transa,
        const at::Tensor& B,
        int64_t ldb,
        bool transb,
        at::Tensor& C,
        int64_t ldc,
        int64_t m,
        int64_t n,
        int64_t k,
        const at::Scalar& alpha,
        const at::Scalar& beta);
void cublas_syrk(
        const at::Tensor& A,
        int64_t lda,
        at::Tensor& C,
        int64_t ldc,
        const at::Scalar& alpha,
        const at::Scalar& beta,
        bool upper,
        bool transpose,
        int64_t n,
        int64_t k);
} // namespace ops
} // namespace falkon

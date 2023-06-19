#include "cublas_bindings.h"
#include "../helpers.h"
#include "cuda_helpers.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace falkon {
namespace ops {


/*
 * TRSM
 */
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
        int ldb) {
    throw std::invalid_argument("scalar_t");
}
template<>
void trsm<double>(
        cublasHandle_t cublas_handle,
        cublasSideMode_t side,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int m,
        int n,
        const double *alpha,
        const double *A,
        int lda,
        double *B,
        int ldb) {
    FLK_CUDABLAS_CHECK(cublasDtrsm(cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}
template<>
void trsm<float>(
        cublasHandle_t cublas_handle,
        cublasSideMode_t side,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int m,
        int n,
        const float *alpha,
        const float *A,
        int lda,
        float *B,
        int ldb) {
    FLK_CUDABLAS_CHECK(cublasStrsm(cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}


/*
 * TRMM
 */
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
        int ldc) {
    throw std::invalid_argument("scalar_t");
}
template<>
void trmm<double>(
        cublasHandle_t cublas_handle,
        cublasSideMode_t side,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int m,
        int n,
        const double *alpha,
        const double *A,
        int lda,
        const double *B,
        int ldb,
        double *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasDtrmm(cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
}
template<>
void trmm<float>(
        cublasHandle_t cublas_handle,
        cublasSideMode_t side,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int m,
        int n,
        const float *alpha,
        const float *A,
        int lda,
        const float *B,
        int ldb,
        float *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasStrmm(cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
}


/*
 * GEMM
 */
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
        int ldc) {
    throw std::invalid_argument("scalar_t");
}
template<>
void gemm<double>(
        cublasHandle_t cublas_handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const double *alpha,
        const double *A,
        int lda,
        const double *B,
        int ldb,
        const double *beta,
        double *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasDgemm(cublas_handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}
template<>
void gemm<float>(
        cublasHandle_t cublas_handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float *alpha,
        const float *A,
        int lda,
        const float *B,
        int ldb,
        const float *beta,
        float *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasSgemm(cublas_handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}


/*
 * SYRK
 */
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
        int ldc) {
    throw std::invalid_argument("scalar_t");
}
template<>
void syrk<double>(
        cublasHandle_t cublas_handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        int n,
        int k,
        const double *alpha,
        const double *A,
        int lda,
        const double *beta,
        double *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasDsyrk(cublas_handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
}
template<>
void syrk<float>(
        cublasHandle_t cublas_handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        int n,
        int k,
        const float *alpha,
        const float *A,
        int lda,
        const float *beta,
        float *C,
        int ldc) {
    FLK_CUDABLAS_CHECK(cublasSsyrk(cublas_handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
}


namespace {

/*
 * Copies
 */
void cublas_2d_copy_to_dev_async (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    FLK_CUDABLAS_CHECK(cublasSetMatrixAsync(
        rows, cols, elemSize,
        host_tensor.data_ptr(),
        lda,
        dev_tensor.data_ptr(),
        ldb,
        torch_stream.stream()
    ));
}

void cublas_2d_copy_to_dev (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb) {
    FLK_CUDABLAS_CHECK(cublasSetMatrix(
        rows, cols, elemSize,
        host_tensor.data_ptr(),
        lda,
        dev_tensor.data_ptr(),
        ldb
    ));
}

void cublas_2d_copy_to_host_async(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    FLK_CUDABLAS_CHECK(cublasGetMatrixAsync(
        rows, cols, elemSize,
        dev_tensor.data_ptr(),
        lda,
        host_tensor.data_ptr(),
        ldb,
        torch_stream.stream()
    ));
}

void cublas_2d_copy_to_host(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb) {
    FLK_CUDABLAS_CHECK(cublasGetMatrix(
        rows, cols, elemSize,
        dev_tensor.data_ptr(),
        lda,
        host_tensor.data_ptr(),
        ldb
    ));
}

/*
 * Torch wrappers for linalg functions
 */

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
        int64_t ldb) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "cublas_trsm", [&]{
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto B_data = B.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        trsm<scalar_t>(handle, side, uplo, trans, diag, m, n, &cast_alpha, A_data, lda, B_data, ldb);
    });
}

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
        int64_t ldc) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "cublas_trmm", [&]{
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto B_data = B.data_ptr<scalar_t>();
        auto C_data = C.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        trmm<scalar_t>(handle, side, uplo, trans, diag, m, n, &cast_alpha, A_data, lda, B_data, ldb, C_data, ldc);
    });
}

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
        const at::Scalar& beta) {
    cublasOperation_t transa_op = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb_op = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "cublas_gemm", [&]{
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto B_data = B.data_ptr<scalar_t>();
        auto C_data = C.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        scalar_t cast_beta = beta.to<scalar_t>();

        gemm<scalar_t>(handle, transa_op, transb_op, m, n, k, &cast_alpha, A_data, lda, B_data, ldb, &cast_beta, C_data, ldc);
    });
}

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
        int64_t k) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "cublas_syrk", [&]{
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto C_data = C.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        scalar_t cast_beta = beta.to<scalar_t>();
        syrk<scalar_t>(handle, uplo, op, n, k, &cast_alpha, A_data, lda, &cast_beta, C_data, ldc);
    });
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_2d_copy_to_dev_async"),
      TORCH_FN(cublas_2d_copy_to_dev_async));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_2d_copy_to_dev"),
      TORCH_FN(cublas_2d_copy_to_dev));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_2d_copy_to_host_async"),
      TORCH_FN(cublas_2d_copy_to_host_async));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_2d_copy_to_host"),
      TORCH_FN(cublas_2d_copy_to_host));

  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_trsm"),
      TORCH_FN(cublas_trsm)
  );

  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_trmm"),
      TORCH_FN(cublas_trmm)
  );

  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_gemm"),
      TORCH_FN(cublas_gemm)
  );

  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cublas_syrk"),
      TORCH_FN(cublas_syrk)
  );
}

} // namespace ops
} // namespace falkon

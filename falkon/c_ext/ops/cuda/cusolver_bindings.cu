#include "cusolver_bindings.h"
#include "../helpers.h"
#include "cuda_helpers.cuh"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


namespace falkon {
namespace ops {

/* POTRF Buffer Size */
template<typename scalar_t>
void potrf_buffersize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        scalar_t* A,
        int lda,
        int* lwork) {
    throw std::invalid_argument("scalar_t");
}

template <>
void potrf_buffersize<float>(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float* A,
        int lda,
        int* lwork) {
    FLK_CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template <>
void potrf_buffersize<double>(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double* A,
        int lda,
        int* lwork) {
    FLK_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

/* POTRF */
template<typename scalar_t>
void potrf(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        scalar_t* A,
        int lda,
        scalar_t* work,
        int lwork,
        int* info) {
    throw std::invalid_argument("scalar_t");
}
template<>
void potrf<float>(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float* A,
        int lda,
        float* work,
        int lwork,
        int* info) {
    FLK_CUSOLVER_CHECK(cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info));
}
template<>
void potrf<double>(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double* A,
        int lda,
        double* work,
        int lwork,
        int* info) {
    FLK_CUSOLVER_CHECK(cusolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, info));
}

namespace {

int64_t cusolver_potrf_buffer_size(
        at::Tensor &A,
        bool upper,
        int64_t n,
        int64_t lda) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    int lwork;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "potrf_buffer_size", [&]{
        cusolverDnHandle_t handle = at::cuda::getCurrentCUDASolverDnHandle();
        potrf_buffersize<scalar_t>(handle, uplo, (int)n, A.data_ptr<scalar_t>(), (int)lda, &lwork);
    });
    return (int64_t)lwork;
}

void cusolver_potrf(
        at::Tensor& A,
        at::Tensor& workspace,
        at::Tensor& info,
        int64_t workspace_size,
        bool upper,
        int64_t n,
        int64_t lda) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "potrf", [&]{
        auto handle = at::cuda::getCurrentCUDASolverDnHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto workspace_data = workspace.data_ptr<scalar_t>();
        potrf<scalar_t>(handle, uplo, (int)n, A_data, (int)lda, workspace_data, (int)workspace_size, info.data_ptr<int>());
    });
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cusolver_potrf_buffer_size"),
      TORCH_FN(cusolver_potrf_buffer_size));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cusolver_potrf"),
      TORCH_FN(cusolver_potrf));
}

} // namespace ops
} // namespace falkon

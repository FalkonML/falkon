#include "cusolver_bindings.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"


template <>
void potrf_buffersize<float>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork
) {
    FLK_CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template <>
void potrf_buffersize<double>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork
) {
    FLK_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template<>
void potrf<float>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* info
) {
    FLK_CUSOLVER_CHECK(cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info));
}

template<>
void potrf<double>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* info
) {
    FLK_CUSOLVER_CHECK(cusolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, info));
}


int cusolver_potrf_buffer_size(const torch::Tensor &A, bool upper, int n, int lda) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    int lwork;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "potrf_buffer_size", [&]{
        cusolverDnHandle_t handle = at::cuda::getCurrentCUDASolverDnHandle();
        potrf_buffersize<scalar_t>(handle, uplo, n, NULL, lda, &lwork);
    });
    return lwork;
}

void cusolver_potrf(const torch::Tensor& A, const torch::Tensor& workspace, const torch::Tensor& info, int workspace_size, bool upper, int n, int lda) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    int* info_data = info.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "potrf", [&]{
        auto handle = at::cuda::getCurrentCUDASolverDnHandle();
        auto A_data = A.data_ptr<scalar_t>();
        auto workspace_data = workspace.data_ptr<scalar_t>();
        potrf<scalar_t>(handle, uplo, n, A_data, lda, workspace_data, workspace_size, info_data);
    });
}

#include "cusolver_bindings.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDASolver.h>
#include <cublas_v2.h>



int cusolver_potrf_buffer_size(const torch::Tensor &A, bool upper, int n, int lda) {
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    int lwork;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "potrf_buffer_size", [&]{
        auto handle = at::cuda::getCurrentCUDASolverDnHandle();
        at::cuda::solver::potrf_buffersize<scalar_t>(handle, uplo, n, NULL, lda, &lwork);
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
        at::cuda::solver::potrf<scalar_t>(handle, uplo, n, A_data, lda, workspace_data, workspace_size, info_data)
    });
}

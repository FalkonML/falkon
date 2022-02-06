#pragma once

#include <torch/extension.h>

#include <cublas_v2.h>
#include <cusolverDn.h>



template<class scalar_t>
void potrf_buffersize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, scalar_t* A, int lda, int* lwork) {
    throw std::invalid_argument("scalar_t");
}
template<>
void potrf_buffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork);
template<>
void potrf_buffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork);


template<class scalar_t>
void potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, scalar_t* A, int lda, scalar_t* work, int lwork, int* info) {
    throw std::invalid_argument("scalar_t");
}
template<>
void potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* info);
template<>
void potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* info);


int cusolver_potrf_buffer_size(const torch::Tensor &A, bool upper, int n, int lda);

void cusolver_potrf(const torch::Tensor& A, const torch::Tensor& workspace, const torch::Tensor& info, int workspace_size, bool upper, int n, int lda);

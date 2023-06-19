#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace falkon {
namespace ops {

template<typename scalar_t>
void potrf_buffersize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    scalar_t* A,
    int lda,
    int* lwork);

template<typename scalar_t>
void potrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    scalar_t* A,
    int lda,
    scalar_t* work,
    int lwork,
    int* info);

} // namespace ops
} // namespace falkon

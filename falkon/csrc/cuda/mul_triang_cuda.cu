#include "mul_triang_cuda.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "utils.cuh"


#define NB 64

template <typename scalar_t>
__global__ void mul_upper_diag(scalar_t *data, const size_t size, const scalar_t mul)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data += i * size;
        const scalar_t *diag_stop = data + i;
        while (data <= diag_stop) {
            *data *= mul;
            data++;
        }
    }
}

template <typename scalar_t>
__global__ void mul_upper(scalar_t *data, const size_t size, const scalar_t mul)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data += i * size;
        const scalar_t *diag_stop = data + i;
        while (data < diag_stop) {
            *data *= mul;
            data++;
        }
    }
}

template <typename scalar_t>
__global__ void mul_lower_diag(scalar_t *data, const size_t size, const scalar_t mul)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data += i * size + i;
        const scalar_t *diag_stop = data + size - i;
        while (data < diag_stop) {
            *data *= mul;
            data++;
        }
    }
}

template <typename scalar_t>
__global__ void mul_lower(scalar_t* __restrict__ data, const size_t size, const scalar_t mul)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data += i * size + i;
        const scalar_t* diag_stop = data + size - i;
        data++; // Avoid touching the diagonal
        while (data < diag_stop) {
            *data *= mul;
            data++;
        }
    }
}


torch::Tensor mul_triang_cuda(torch::Tensor &A, const bool upper, const bool preserve_diag, const double multiplier) {
    CHECK_CUDA(A);
    bool bupper = upper;
    if (A.stride(0) != 1) {
        bupper = !bupper;
    }

    const auto nx = A.size(0);
    const auto scalar_type = A.scalar_type();
    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
    const scalar_t mul = (scalar_t)multiplier;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::DeviceGuard g(A.device());
    if (bupper && preserve_diag) {  // U, preserve
        mul_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else if (bupper) {             // U, no-preserve
        mul_upper_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else if (preserve_diag) {     // L, preserve
        mul_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else {                        // L, no-preserve
        mul_lower_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    }
    });
    return A;
}


#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "../helpers.h"

namespace falkon {
namespace ops {
namespace {

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


at::Tensor mul_triang_kernel(at::Tensor &mat, const double multiplier, const bool upper, const bool preserve_diag) {
    CHECK_CUDA(mat);
    TORCH_CHECK(mat.size(0) == mat.size(1), "Input matrix must be square.");

    const bool bupper = is_fortran_contig(mat) ? upper : !upper;
    const int64_t nx = mat.size(0);
    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "dispatch_mul_triang", [&] {
    const scalar_t mul = (scalar_t)multiplier;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::DeviceGuard g(mat.device());
    if (bupper && preserve_diag) {  // U, preserve
        mul_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(mat.data_ptr<scalar_t>(), nx, mul);
    } else if (bupper) {             // U, no-preserve
        mul_upper_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(mat.data_ptr<scalar_t>(), nx, mul);
    } else if (preserve_diag) {     // L, preserve
        mul_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(mat.data_ptr<scalar_t>(), nx, mul);
    } else {                        // L, no-preserve
        mul_lower_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(mat.data_ptr<scalar_t>(), nx, mul);
    }
    });
    return mat;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::mul_triang"),
      TORCH_FN(mul_triang_kernel));
}

} // namespace ops
} // namespace falkon

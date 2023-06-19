#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "../helpers.h"


namespace falkon {
namespace ops {
namespace {

#define NB 64

/*
  Matrix is size * size (no support for different size than stride).
  Columns are contiguous.
  The size * size grid is subdivided into NB * size blocks (of rows).
  Each block has NB threads, so each thread copies one row into one
  column (transpose).
  Not a particularly efficient implementation!
*/
template <typename scalar_t>
__global__ void copy_simple_kernel_lower(
        scalar_t* __restrict__ data,
        const size_t size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int col_pos = i * size;
        for (int row_pos = i; row_pos < i + i * size; row_pos += size) {
            data[col_pos] = data[row_pos];
            col_pos++;
        }
    }
}

// Same as the _lower version, but we copy dataT to data instead!
template <typename scalar_t>
__global__ void copy_simple_kernel_upper(
        scalar_t* __restrict__ data,
        const size_t size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int col_pos = i * size;
        for (int row_pos = i; row_pos < i + i * size; row_pos += size) {
            data[row_pos] = data[col_pos];
            col_pos++;
        }
    }
}

at::Tensor copy_triang_kernel(
        at::Tensor &A,
        const bool upper) {
    CHECK_CUDA(A);
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square 2D matrix.");

    // Transpose matrix, and flip upper if matrix is C-contiguous.
    const bool fContig = is_fortran_contig(A);
    if (!fContig)
      A = at::transpose(A, 0, 1);
    const bool bupper = fContig ? upper : !upper;
    const int64_t nx = A.size(0);
    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "dispatch_copy_triang", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        if (bupper) {
            copy_simple_kernel_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
        } else {
            copy_simple_kernel_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
        }
    });

    if (!fContig)
        A = at::transpose(A, 0, 1);
    return A;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::copy_triang"),
      TORCH_FN(copy_triang_kernel));
}

} // namespace ops
} // namespace falkon

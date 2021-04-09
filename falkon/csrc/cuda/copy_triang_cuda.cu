#include "copy_triang_cuda.h"
#include "utils.cuh"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>


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
__global__ void copy_simple_kernel_lower(scalar_t* __restrict__ data, const size_t size)
{
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
__global__ void copy_simple_kernel_upper(scalar_t* __restrict__ data, const size_t size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int col_pos = i * size;
        for (int row_pos = i; row_pos < i + i * size; row_pos += size) {
            data[row_pos] = data[col_pos];
            col_pos++;
        }
    }
}


torch::Tensor cuda_copy_triang(torch::Tensor &A, const bool upper) {
    CHECK_CUDA(A);

    bool needs_transpose = false;
    bool bupper = upper;
    if (A.stride(0) != 1) {
        // Not F-contig (assume C-contig)
        A = torch::transpose(A, 0, 1);
        bupper = !bupper;
        needs_transpose = true;
    }

    const auto nx = A.size(0);
    const auto ny = A.size(1);
    const auto scalar_type = A.scalar_type();

    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    /* Run CUDA kernel */
    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        if (bupper) {
            copy_simple_kernel_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
        } else {
            copy_simple_kernel_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
        }
    });

    if (needs_transpose) {
        A = torch::transpose(A, 0, 1);
    }
    return A;
}

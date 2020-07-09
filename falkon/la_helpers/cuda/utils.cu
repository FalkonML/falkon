#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>


#define NB 64
#define TILE_DIM 32


/*
  Matrix is size * size (no support for different size than stride).
  Columns are contiguous.
  The size * size grid is subdivided into NB * size blocks (of rows).
  Each block has NB threads, so each thread copies one row into one
  column (transpose).
  Not a particularly efficient implementation!
*/
template <typename scalar_t>
__global__ void copy_simple_kernel_lower(scalar_t *data, const size_t size)
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
__global__ void copy_simple_kernel_upper(scalar_t *data, const size_t size)
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
__global__ void mul_lower(scalar_t *data, const size_t size, const scalar_t mul)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data += i * size + i;
        const scalar_t *diag_stop = data + size - i;
        data++; // Avoid touching the diagonal
        while (data < diag_stop) {
            *data *= mul;
            data++;
        }
    }
}


template<typename scalar_t>
__global__
void matrix_transpose(scalar_t * out, const scalar_t * in, const unsigned dim0, const unsigned dim1)
{
    __shared__ scalar_t shrdMem[TILE_DIM][TILE_DIM+1];

    unsigned lx = threadIdx.x;
    unsigned ly = threadIdx.y;

    unsigned gx = lx + blockDim.x * blockIdx.x;
    unsigned gy = ly + TILE_DIM   * blockIdx.y;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy + repeat;
        if (gx < dim0 && gy_ < dim1)
            shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
    }
    __syncthreads();

    gx = lx + blockDim.x * blockIdx.y;
    gy = ly + TILE_DIM   * blockIdx.x;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy + repeat;
        if (gx < dim1 && gy_ < dim0)
            out[gy_ * dim0 + gx] = shrdMem[lx][ly + repeat];
    }
}


int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}


torch::Tensor cuda_copy_triang(torch::Tensor &A, bool upper) {
    if (!A.is_cuda()) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }

    bool needs_transpose = false;
    if (A.stride(0) != 1) {
        // Not F-contig (assume C-contig)
        A = torch::transpose(A, 0, 1);
        upper = !upper;
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
    if (upper) {
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

torch::Tensor cuda_mul_triang(torch::Tensor &A, bool upper, const bool preserve_diag, const double multiplier) {
    if (!A.is_cuda()) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }
    if (A.stride(0) != 1) {
        upper = !upper;
    }

    const auto nx = A.size(0);
    const auto scalar_type = A.scalar_type();
    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
    const scalar_t mul = (scalar_t)multiplier;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::DeviceGuard g(A.device());
    if (upper && preserve_diag) {  // U, preserve
        mul_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else if (upper) {            // U, no-preserve
        mul_upper_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else if (preserve_diag) {    // L, preserve
        mul_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    } else {                       // L, no-preserve
        mul_lower_diag<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx, mul);
    }
    });
    return A;
}

torch::Tensor cuda_transpose(torch::Tensor &input, torch::Tensor &output) {
    if (!input.is_cuda())
        AT_ERROR("Input A must be a CUDA tensor.");
    if (!output.is_cuda())
        AT_ERROR("Input A must be a CUDA tensor.");
    if (input.size() != output.size())
        AT_ERROR("Input and output matrices must be of the same size.");

    const auto nx = A.size(0);
    const auto ny = A.size(1);
    const auto scalar_type = A.scalar_type();

    const dim3 dimGrid(ceildiv(nx, TILE_DIM), ceildiv(ny, TILE_DIM), 1);
    const dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::DeviceGuard g(A.device());
    matrix_transpose<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), nx, ny);
    });
    return output;
}
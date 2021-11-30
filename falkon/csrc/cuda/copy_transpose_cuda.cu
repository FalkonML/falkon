#include "copy_transpose_cuda.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"


#define NB 32
#define BLOCK_ROWS 8

template<typename scalar_t>
__global__
void matrix_transpose_f(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, const unsigned dim0, const unsigned dim1)
{
    // https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
    // https://arrayfire.com/cuda-optimization-tips-for-matrix-transpose-in-real-world-applications/
    __shared__ scalar_t shrdMem[NB][NB+1];

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;

    unsigned gx = lx + NB * blockIdx.x;
    unsigned gy = ly + NB * blockIdx.y;

    #pragma unroll
    for (unsigned repeat = 0; repeat < NB; repeat += blockDim.y) {
        unsigned gy_ = gy + repeat;
        if (gx < dim0 && gy_ < dim1) {
            shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
        }
    }
    __syncthreads();

    gx = lx + NB * blockIdx.y;
    gy = ly + NB * blockIdx.x;

    #pragma unroll
    for (unsigned repeat = 0; repeat < NB; repeat += blockDim.y) {
        unsigned gy_ = gy + repeat;
        if (gx < dim1 && gy_ < dim0) {
            out[gy_ * dim1 + gx] = shrdMem[lx][ly + repeat];
        }
    }
}


template<typename scalar_t>
__global__
void matrix_transpose_c(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, const unsigned dim0, const unsigned dim1)
{
    // https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
    // https://arrayfire.com/cuda-optimization-tips-for-matrix-transpose-in-real-world-applications/
    __shared__ scalar_t shrdMem[NB][NB+1];

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;

    unsigned gx = lx + NB * blockIdx.x;
    unsigned gy = ly + NB * blockIdx.y;

    #pragma unroll
    for (unsigned repeat = 0; repeat < NB; repeat += blockDim.x) {
        unsigned gx_ = gx + repeat;
        if (gx_ < dim0 && gy < dim1) {
            shrdMem[lx + repeat][ly] = in[gx_ * dim1 + gy];
        }
    }
    __syncthreads();

    gx = lx + NB * blockIdx.y;
    gy = ly + NB * blockIdx.x;

    #pragma unroll
    for (unsigned repeat = 0; repeat < NB; repeat += blockDim.x) {
        unsigned gx_ = gx + repeat;
        if (gx_ < dim1 && gy < dim0) {
            out[gx_ * dim0 + gy] = shrdMem[ly][lx + repeat];
        }
    }
}


torch::Tensor copy_transpose_cuda(const torch::Tensor &input, torch::Tensor &output) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    TORCH_CHECK(input.size(0) == output.size(1) && input.size(1) == output.size(0),
                "Input and output matrices shapes must be consistent.");
    // TODO: Check strides are consistent

    const int64_t nx = input.size(0), ny = input.size(1);
    const bool fortran_contig = is_fortran_contig(input);

    const dim3 dimGrid(ceildiv(nx, NB), ceildiv(ny, NB), 1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dispatch_copy_transpose", [&] {
        at::DeviceGuard g(input.device());
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        if (fortran_contig) {
            const dim3 dimBlock(NB, BLOCK_ROWS, 1);
            matrix_transpose_f<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), nx, ny);
        } else {
            const dim3 dimBlock(BLOCK_ROWS, NB, 1);
            matrix_transpose_c<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), nx, ny);
        }
    });
    return output;
}

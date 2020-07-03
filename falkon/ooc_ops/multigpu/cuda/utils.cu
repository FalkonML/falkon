#include <thread>
#include <stdio.h>

#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>


template <typename scalar_t>
__global__ void transposeCoalesced(scalar_t *data)
{
    // https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
    // blockDim.x == TILE_DIM
    // blockDim.y == BLOCK_ROWS
    // Copy idata into shared memory (transposing)
    // Padding + 1 to avoid share-memory-bank conflicts
    __shared__ scalar_t tile[blockDim.x][blockDim.x + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int j = 0; j < blockDim.x; j += blockDim.y) {
        tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
    }

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;  // transpose block offset
    y = blockIdx.x * blockDim.x + threadIdx.y;

    for (int j = 0; j < blockDim.x; j += blockDim.y) {
        if ((y + j)*width < x) {
            data[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


torch::Tensor cuda_copy_triang(torch::Tensor &A, bool upper) {
    if (!A.is_cuda) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }

    bool needs_transpose = false;
    if (A.stride(0) != 1) {
        // Not F-contig (assume C-contig)
        A = A.transpose();
        upper = !upper;
        needs_transpose = true;
    }

    const int nx = A.size(0);
    const int ny = A.size(1);
    const auto scalar_type = A.scalar_type();

    const int tile_dim = 32;
    const int block_rows = 8;

    dim3 dimGrid(nx/tile_dim, ny/tile_dim, 1);
    dim3 dimBlock(tile_dim, block_rows, 1);

    /* Run CUDA kernel */
    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
        scalar_t *data = A.data_ptr<scalar_t>();
        transposeCoalesced<<<dimGrid, dimBlock>>>(data);
    }

    if (needs_transpose) {
        A = A.transpose();
    }
    return A;
}

torch::Tensor mul_triang(torch::Tensor &A, bool upper, bool preserve_diag, torch::Tensor &multiplier) {

}
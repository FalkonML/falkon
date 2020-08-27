#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>


#include "lauum.cuh"
int ceildiv(int dividend, int divisor);


#define BLOCK_SIZE 32
//#define DEBUG

__device__ int2 tri_index(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        linear_index - row * (row + 1) / 2,
        row
    );
}

template<typename scalar_t>
__global__
void lower_cuda_lauum_ker(const scalar_t* __restrict__ in,
                          scalar_t *out,
                          const int size,
                          const int in_stride,
                          const int out_stride,
                          const int grid_size) {
    // Determine the triangular tile of the output (0 indexed)
    const int2 tile_pos = tri_index(blockIdx.x);
//    const int element = blockIdx.x;
//    const int tile_row = (int)((-1 + sqrt((double)(8*element + 1))) / 2.0);
//    const int tile_col = element - tile_row * (tile_row + 1) / 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // A_col is the global column of the current thread for the A tile (transposed)
    const int A_col = tile_pos.y * BLOCK_SIZE + tx;
    // B_col is the global column of the current thread for the B tile (not transposed)
    const int B_col = tile_pos.x * BLOCK_SIZE + tx;

    // Initialize shared mem
    __shared__ scalar_t A_tile[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ scalar_t B_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (int tile_i = tile_pos.y; tile_i < grid_size; tile_i++) {
        // i is the row position of this thread within tile-rows
        int i = tile_i * BLOCK_SIZE + ty;

        // Copy item input[i, row].T and input[i, col] to shared memory
        A_tile[ty][tx] = 0;
        B_tile[ty][tx] = 0;
        if (i < size && A_col < size && A_col <= i) {
            A_tile[ty][tx] = in[i + in_stride * A_col];
        }
        if (i < size && B_col < size && B_col <= i) {
            B_tile[ty][tx] = in[i + in_stride * B_col];
        }
        __syncthreads();
        
        #ifdef DEBUG
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - A[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, ty, tx, A_tile[ty][tx]);
        __syncthreads();
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - B[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, ty, tx, B_tile[ty][tx]);
        __syncthreads();
        #endif

        // Compute
        for (int k = 0; k < BLOCK_SIZE; k++) {
            accumulator = accumulator + A_tile[k][ty] * B_tile[k][tx];
        }
        __syncthreads();
    }
    // Write-back
    const int col = tile_pos.x * BLOCK_SIZE + tx;
    const int row = tile_pos.y * BLOCK_SIZE + ty;
    if (row >= col && col < size && row < size) {
        out[row + col * out_stride] = accumulator;
    }
}

torch::Tensor lauum_lower(torch::Tensor &input, torch::Tensor &output) {
    // TODO: Consistency checks

    const auto scalar_type = input.scalar_type();
    const auto size = input.size(0);
    const auto in_stride = input.stride(1);
    const auto out_stride = output.stride(1);

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const int grid_height = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(input.device());
        lower_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
    });
    return output;
}


int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}

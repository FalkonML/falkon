#include "lauum.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "utils.cuh"


#define BLOCK_SIZE 32
//#define DEBUG


template<typename scalar_t>
__global__
void upper_cuda_lauum_ker(const scalar_t* __restrict__ in,
                          scalar_t* __restrict__ out,
                          const int size,
                          const int in_stride,
                          const int out_stride,
                          const int grid_size) {
    const int2 tile_pos = tri_index_upper(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // tx and ty are inverted (i.e. tx goes on along the rows,
    // ty along the columns). This allows coalesced store in the
    // write-back phase
    const int A_row = tile_pos.y * BLOCK_SIZE + tx;
    const int B_row = tile_pos.x * BLOCK_SIZE + tx;

    // Initialize shared mem
    __shared__ scalar_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    // The first dimension of the B_tile needs to be increased to prevent bank
    // conflicts in B_tile load.
    __shared__ scalar_t B_tile[(BLOCK_SIZE + 1) * BLOCK_SIZE];

    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (int tile_i = tile_pos.x; tile_i < grid_size; tile_i++) {
        const int i = tile_i * BLOCK_SIZE + ty; 
	const int i_pos = i * in_stride;

        // Copy item input[row, i] and input[col, i].T to shared memory
        A_tile[ty * BLOCK_SIZE + tx] = 0;
        B_tile[tx * (BLOCK_SIZE + 1) + ty] = 0;
	if (i < size && A_row <= i) {
	    A_tile[ty * BLOCK_SIZE + tx] = in[A_row + i_pos];
	}
	if (i < size && B_row <= i) {
            B_tile[tx * (BLOCK_SIZE + 1) + ty] = in[B_row + i_pos];
	}
        __syncthreads();

	#ifdef DEBUG
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - A[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, tx, ty, A_tile[tx][ty]);
        __syncthreads();
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - B[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, tx, ty, B_tile[tx][ty]);
        __syncthreads();
	#endif

        // Compute
        for (int k = 0; k < BLOCK_SIZE; k++) {
	    // Both accesses to A, B are done to prevent bank conflicts.
	    // In practice we need to avoid stuff like A[tx][k] where tx is on the first dimension.
            accumulator = accumulator + A_tile[k * BLOCK_SIZE + tx] * B_tile[ty * (BLOCK_SIZE + 1) + k];
        }
        __syncthreads();
    }
    // Write-back
    const int col = tile_pos.x * BLOCK_SIZE + ty;
    const int row = tile_pos.y * BLOCK_SIZE + tx;
    if (row <= col && col < size && row < size) {
        out[row + col * out_stride] = accumulator;
    }
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
    const int2 tile_pos = tri_index_lower(blockIdx.x);
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
        if (i < size && A_col <= i) {
            A_tile[ty][tx] = in[i + in_stride * A_col];
        }
        if (i < size && B_col <= i) {
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


torch::Tensor lauum_cuda(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb, const bool lower) {
    // TODO: Consistency checks
    const auto scalar_type = A.scalar_type();
    const auto size = n;
    const auto in_stride = lda;
    const auto out_stride = ldb;

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const int grid_height = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::DeviceGuard g(A.device());
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	if (lower) {
            lower_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
	}
       	else {
            upper_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
	}
    });
    return B;
}


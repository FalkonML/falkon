#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>


#define NB 64
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define VEC_MUL_TRIANG_BLOCK_DIM 32


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


__device__ int2 tri_index_lower(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        linear_index - row * (row + 1) / 2,
        row
    );
}

__device__ int2 tri_index_upper(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        row,
        linear_index - row * (row + 1) / 2
    );
}

/* upper = 1, side = 1 (from top) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v1(scalar_t* __restrict__ mat, const scalar_t* __restrict__ vec, const int mat_stride, const int mat_size) {
    const int2 tile_pos = tri_index_upper(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Check if thread is out of bounds
    const int gx = tile_pos.x * VEC_MUL_TRIANG_BLOCK_DIM + ty;
    const int gy = tile_pos.y * VEC_MUL_TRIANG_BLOCK_DIM + tx;
    if (gy > gx || gx >= mat_size || gy >= mat_size) {
        return;
    }

    // Copy global to register mem
    scalar_t val = mat[gx * mat_stride + gy];
    const scalar_t mul = vec[gx];
    // Calc
    val *= mul;
    // Copy back
    mat[gx * mat_stride + gy] = val;
}

/* upper = 1, side = 0 (from left) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v2(scalar_t* __restrict__ mat, const scalar_t* __restrict__ vec, const int mat_stride, const int mat_size) {
    const int2 tile_pos = tri_index_upper(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Check if thread is out of bounds
    const int gx = tile_pos.x * VEC_MUL_TRIANG_BLOCK_DIM + ty;
    const int gy = tile_pos.y * VEC_MUL_TRIANG_BLOCK_DIM + tx;
    if (gy > gx || gx >= mat_size || gy >= mat_size) {
        return;
    }

    // Copy global to register mem
    scalar_t val = mat[gx * mat_stride + gy];
    const scalar_t mul = vec[gy];
    // Calc
    val *= mul;
    // Copy back
    mat[gx * mat_stride + gy] = val;
}

/* upper = 0, side = 1 (from top) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v3(scalar_t* __restrict__ mat, const scalar_t* __restrict__ vec, const int mat_stride, const int mat_size) {
    const int2 tile_pos = tri_index_lower(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Check if thread is out of bounds
    const int gx = tile_pos.x * VEC_MUL_TRIANG_BLOCK_DIM + ty;
    const int gy = tile_pos.y * VEC_MUL_TRIANG_BLOCK_DIM + tx;
    if (gy < gx || gx >= mat_size || gy >= mat_size) {
        return;
    }

    // Copy global to register mem
    scalar_t val = mat[gx * mat_stride + gy];
    const scalar_t mul = vec[gx];
    // Calc
    val *= mul;
    // Copy back
    mat[gx * mat_stride + gy] = val;
}

/* upper = 0, side = 0 (from left) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v4(scalar_t* __restrict__ mat, const scalar_t* __restrict__ vec, const int mat_stride, const int mat_size) {
    const int2 tile_pos = tri_index_lower(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Check if thread is out of bounds
    const int gx = tile_pos.x * VEC_MUL_TRIANG_BLOCK_DIM + ty;
    const int gy = tile_pos.y * VEC_MUL_TRIANG_BLOCK_DIM + tx;
    if (gy < gx || gx >= mat_size || gy >= mat_size) {
        return;
    }

    // Copy global to register mem
    scalar_t val = mat[gx * mat_stride + gy];
    const scalar_t mul = vec[gy];
    // Calc
    val *= mul;
    // Copy back
    mat[gx * mat_stride + gy] = val;
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


template<typename scalar_t>
__global__
void matrix_transpose_f(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, const unsigned dim0, const unsigned dim1)
{
    // https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
    // https://arrayfire.com/cuda-optimization-tips-for-matrix-transpose-in-real-world-applications/
    __shared__ scalar_t shrdMem[TILE_DIM][TILE_DIM+1];

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;

    unsigned gx = lx + TILE_DIM * blockIdx.x;
    unsigned gy = ly + TILE_DIM * blockIdx.y;

    #pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy + repeat;
        if (gx < dim0 && gy_ < dim1) {
            shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
        }
    }
    __syncthreads();

    gx = lx + TILE_DIM * blockIdx.y;
    gy = ly + TILE_DIM * blockIdx.x;

    #pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
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
    __shared__ scalar_t shrdMem[TILE_DIM][TILE_DIM+1];

    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;

    unsigned gx = lx + TILE_DIM * blockIdx.x;
    unsigned gy = ly + TILE_DIM * blockIdx.y;

    #pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.x) {
        unsigned gx_ = gx + repeat;
        if (gx_ < dim0 && gy < dim1) {
            shrdMem[lx + repeat][ly] = in[gx_ * dim1 + gy];
        }
    }
    __syncthreads();

    gx = lx + TILE_DIM * blockIdx.y;
    gy = ly + TILE_DIM * blockIdx.x;

    #pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.x) {
        unsigned gx_ = gx + repeat;
        if (gx_ < dim1 && gy < dim0) {
            out[gx_ * dim0 + gy] = shrdMem[ly][lx + repeat];
        }
    }
}

int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}


torch::Tensor cuda_vec_mul_triang(torch::Tensor &A, torch::Tensor &v, bool upper, int side) {
    if (!A.is_cuda())
        AT_ERROR("Input A must be a CUDA tensor.");
    if (!v.is_cuda())
        AT_ERROR("Input v must be a CUDA tensor.");
    if (device_of(v) != device_of(A))
        AT_ERROR("Inputs A, v must be on the same CUDA device.");
    if (A.size(0) != A.size(1))
        AT_ERROR("Input A must be square.");
    if (A.size(0) != v.size(0))
        AT_ERROR("Input v must be of the same dimension as matrix A.");

    int mat_stride = A.stride(1);
    const auto mat_size = A.size(0);
    const auto scalar_type = A.scalar_type();
    // Check matrix contiguity
    bool fortran_contig = false;
    if (A.stride(0) == 1)
        fortran_contig = true;
    // Flip operation if C-contiguous
    bool bside = (bool)side;
    bool bupper = upper;
    if (!fortran_contig) {
        bupper = !upper;
        bside = !bside;
        mat_stride = A.stride(0);
    }

    const int grid_height = ceildiv(mat_size, VEC_MUL_TRIANG_BLOCK_DIM);
    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(VEC_MUL_TRIANG_BLOCK_DIM, VEC_MUL_TRIANG_BLOCK_DIM);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch_vec_mul_triang", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        // Choose correct kernel
        if (bupper && bside)
            vec_mul_triang_kernel_v1<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (bupper && !bside)
            vec_mul_triang_kernel_v2<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (!bupper && bside)
            vec_mul_triang_kernel_v3<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (!bupper && !bside)
            vec_mul_triang_kernel_v4<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                A.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), mat_stride, mat_size);
    });
    return A;
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
        AT_ERROR("Input must be a CUDA tensor.");
    if (!output.is_cuda())
        AT_ERROR("Output must be a CUDA tensor.");
    if (input.size(0) != output.size(1) || input.size(1) != output.size(0))
        AT_ERROR("Input and output matrices must be of the same size.");
    // TODO: Check strides are consistent

    const auto nx = input.size(0);
    const auto ny = input.size(1);
    const auto scalar_type = input.scalar_type();
    bool fortran_contig = false;
    if (input.stride(0) == 1) {
        fortran_contig = true;
    }

    const dim3 dimGrid(ceildiv(nx, TILE_DIM), ceildiv(ny, TILE_DIM), 1);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(input.device());
        if (fortran_contig) {
            const dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            matrix_transpose_f<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), nx, ny);
        } else {
            const dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
            matrix_transpose_c<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), nx, ny);
        }
    });
    return output;
}

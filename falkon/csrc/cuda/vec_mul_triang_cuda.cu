#include "vec_mul_triang_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"


#define NB 32

/* upper = 1, side = 1 (from top) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v1(scalar_t* __restrict__ mat, const scalar_t* __restrict__ vec, const int mat_stride, const int mat_size) {
    const int2 tile_pos = tri_index_upper(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Check if thread is out of bounds
    const int gx = tile_pos.x * NB + ty;
    const int gy = tile_pos.y * NB + tx;
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
    const int gx = tile_pos.x * NB + ty;
    const int gy = tile_pos.y * NB + tx;
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
    const int gx = tile_pos.x * NB + ty;
    const int gy = tile_pos.y * NB + tx;
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
    const int gx = tile_pos.x * NB + ty;
    const int gy = tile_pos.y * NB + tx;
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


torch::Tensor vec_mul_triang_cuda(torch::Tensor &A, torch::Tensor &v, const bool upper, const int side) {
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
    // Flip operation if C-contiguous
    const bool fortran_contig = is_fortran_contig(A);
    bool bside = (bool)side;
    bool bupper = upper;
    if (!fortran_contig) {
        bupper = !upper;
        bside = !bside;
        mat_stride = A.stride(0);
    }

    const int grid_height = ceildiv(mat_size, NB);
    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(NB, NB);

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

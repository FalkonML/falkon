#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "../helpers.h"
#include "cuda_helpers.cuh"

namespace falkon {
namespace ops {
namespace {

#define NB 32

/* upper = 1, side = 1 (from top) */
template <typename scalar_t>
__global__ void vec_mul_triang_kernel_v1(
        scalar_t* __restrict__ mat,
        const scalar_t* __restrict__ vec,
        const int mat_stride,
        const int mat_size) {
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

at::Tensor vec_mul_triang_kernel(
        at::Tensor &mat,
        const at::Tensor &multiplier_vec,
        const bool upper,
        const bool side) {
    CHECK_CUDA(mat);
    CHECK_CUDA(multiplier_vec);
    TORCH_CHECK(device_of(multiplier_vec) == device_of(mat), "mat and multiplier_vec must be on the same CUDA device.");
    TORCH_CHECK(mat.size(0) == mat.size(1), "mat must be a square 2D matrix.");
    TORCH_CHECK(mat.size(0) == multiplier_vec.size(0), "multiplier_vec must be a vector with size matching mat.");

    int64_t mat_stride = mat.stride(1), mat_size = mat.size(0);
    bool bside = side, bupper = upper;
    // Flip operation if C-contiguous
    if (!is_fortran_contig(mat)) {
        bupper = !bupper;
        bside = !bside;
        mat_stride = mat.stride(0);
    }

    const int grid_height = ceildiv(mat_size, NB);
    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(NB, NB);

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "dispatch_vec_mul_triang", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(mat.device());
        // Choose correct kernel
        if (bupper && bside)
            vec_mul_triang_kernel_v1<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                mat.data_ptr<scalar_t>(), multiplier_vec.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (bupper && !bside)
            vec_mul_triang_kernel_v2<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                mat.data_ptr<scalar_t>(), multiplier_vec.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (!bupper && bside)
            vec_mul_triang_kernel_v3<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                mat.data_ptr<scalar_t>(), multiplier_vec.data_ptr<scalar_t>(), mat_stride, mat_size);
        else if (!bupper && !bside)
            vec_mul_triang_kernel_v4<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
                mat.data_ptr<scalar_t>(), multiplier_vec.data_ptr<scalar_t>(), mat_stride, mat_size);
    });
    return mat;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::vec_mul_triang"),
      TORCH_FN(vec_mul_triang_kernel));
}

} // namespace ops
} // namespace falkon


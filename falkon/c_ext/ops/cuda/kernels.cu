/*
 * copyright PyTorch authors 2026.
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/Exceptions.h>

#include <math.h>
#include <iostream>

#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/macros/Macros.h>

#include "../helpers.h"

namespace falkon {
namespace ops {
namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kCUDANumWarpsPerBlock = kCUDANumThreads / 32;


template <typename scalar_t>
struct DistReduceOp {
    __forceinline__ __device__ scalar_t combine(scalar_t a, scalar_t b) const {
        a += b;
        return a;
    }

    __forceinline__ __device__ scalar_t warp_shfl_down(scalar_t data, int offset) const {
        return WARP_SHFL_DOWN(data, offset);
    }
};

template <typename scalar_t>
__global__ static void manhattan_kernel_cuda_impl_C(
    scalar_t* __restrict__ result,  // [d, r1, r2]
    const scalar_t* __restrict__ x1,  // [d, r1, m]
    const scalar_t* __restrict__ x2,  // [d, r2, m]
    const int64_t r2,
    const int64_t m,
    const int64_t r1,
    const int64_t result_stride_0,
    const int64_t result_stride_1) {
  // one warp per output entry
  // each block has e.g 256 threads and 8 warps
  // hence processes 8 output entries
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  const int64_t pair = static_cast<int64_t>(blockIdx.x) * kCUDANumWarpsPerBlock + warp_id;
  const int64_t batch_id = pair / (r1 * r2);
  const int64_t batch_offset = pair % (r1 * r2);
  const int64_t pair_1 = batch_offset / r1;
  const int64_t pair_2 = pair % r1;
  if (pair >= r1 * r2) {
    return;
  }

  const int64_t i = pair / r2;
  const int64_t j = pair % r2;

  const scalar_t* a = x1 + i * m;
  const scalar_t* b = x2 + j * m;

  scalar_t agg = scalar_t(0);
  for (int64_t k = lane_id; k < m; k += 32) {
    agg += std::abs(a[k] - b[k]);
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    agg += __shfl_down_sync(0xffffffff, agg, offset);
  }
  if (lane_id == 0) {
    result[batch_id * result_stride_0 + pair_1 * result_stride_1] = agg;
  }
}

template <typename scalar_t>
__global__ static void manhattan_kernel_cuda_impl_F(
    scalar_t* result,
    const scalar_t* x1,
    const scalar_t* x2,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size,
    const int64_t d,
    const int64_t r1) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;

  const scalar_t* a = x1 + l + i * d;
  const scalar_t* b = x2 + l + j * d;

  scalar_t agg = 0;

  for (int64_t q = threadIdx.x; q < m * d; q += blockDim.x) {
    const int64_t p = q % d;
    const int64_t q_m = q / d;

    agg += std::abs(a[q_m * d * r1 + p] - b[q_m * d * r2 + p]);
  }

  __shared__ scalar_t agg_smem[kCUDANumThreads];
  scalar_t agg_init{0.0};

  agg = at::native::cuda_utils::BlockReduce(agg, DistReduceOp<scalar_t>{}, agg_init, agg_smem);

  if (threadIdx.x == 0) {
    result[l + i * d + j * d * r1] = agg;
  }
}

template <typename scalar_t>
__global__ static void manhattan_kernel_cuda_impl_strided(
    scalar_t * result,
    const scalar_t * x1,
    const scalar_t * x2,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t s1_d,
    const int64_t s1_r,
    const int64_t s1_m,
    const int64_t s2_d,
    const int64_t s2_r,
    const int64_t s2_m,
    const int64_t sr_d,
    const int64_t sr_r,
    const int64_t sr_j) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const scalar_t *a = x1 + l * s1_d + i * s1_r;
  const scalar_t *b = x2 + l * s2_d + j * s2_r;

  scalar_t agg = 0.0;
  for (int64_t q = threadIdx.x; q < m; q += stride) {
    agg += std::abs(a[q * s1_m] - b[q * s2_m]);
  }

  __shared__ scalar_t agg_smem[kCUDANumThreads];
  scalar_t agg_init{0.0};
  agg = at::native::cuda_utils::BlockReduce(agg, DistReduceOp<scalar_t>{}, agg_init, agg_smem);

  if (threadIdx.x == 0) {
    result[l * sr_d + i * sr_r + j * sr_j] = agg;
  }
}

bool is_fortran_contiguous(const at::Tensor& x) {
    return x.stride(-2) == 1 && x.stride(-1) == x.size(-2);
}

bool is_c_contiguous(const at::Tensor& x) {
    return x.stride(-1) == 1 && x.stride(-2) == x.size(-1);
}

at::Tensor manhattan_kernel_impl(at::Tensor& result, const at::Tensor& x1, const at::Tensor& x2) {
  CHECK_CUDA(result);
  CHECK_CUDA(x1);
  CHECK_CUDA(x2);
  const int64_t d = x1.size(0);
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;

  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cdist_cuda", [&] {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    if (is_fortran_contiguous(x1) && is_fortran_contiguous(x2) && is_fortran_contiguous(result)) {
      const dim3 grid(result.numel());
      const dim3 block(kCUDANumThreads);
      manhattan_kernel_cuda_impl_F<scalar_t><<<grid, block, 0, stream.stream()>>>(
        result.mutable_data_ptr<scalar_t>(), 
        x1.const_data_ptr<scalar_t>(), 
        x2.const_data_ptr<scalar_t>(),
        r2, 
        m, 
        r_size, 
        l1_size, 
        l2_size,
        d,
        r1
      );
    } else if (is_c_contiguous(x1) && is_c_contiguous(x2) && result.stride(-1) == 1) {
      const dim3 block(kCUDANumThreads);
      const dim3 grid((r_size + kCUDANumWarpsPerBlock - 1) / kCUDANumWarpsPerBlock);
      std::cout << "Block dimensions: (" 
                << block.x << ")" << std::endl;
      std::cout << "Grid dimensions: (" 
                << grid.x << ")" << std::endl;
      manhattan_kernel_cuda_impl_C<scalar_t><<<grid, block, 0, stream.stream()>>>(
        result.mutable_data_ptr<scalar_t>(), 
        x1.const_data_ptr<scalar_t>(), 
        x2.const_data_ptr<scalar_t>(),
        r2, 
        m, 
        r1,
        result.stride(0),
        result.stride(1)
    );
    } else {
      const dim3 grid(result.numel());
      const dim3 block(kCUDANumThreads);

      manhattan_kernel_cuda_impl_strided<scalar_t><<<grid, block, 0, stream.stream()>>>(
        result.mutable_data_ptr<scalar_t>(), 
        x1.const_data_ptr<scalar_t>(), 
        x2.const_data_ptr<scalar_t>(),
        r2,
        m,
        r_size,
        x1.stride(0),
        x1.stride(1),
        x1.stride(2),
        x2.stride(0),
        x2.stride(1),
        x2.stride(2),
        result.stride(0),
        result.stride(1),
        result.stride(2)
      );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::manhattan_dist"),
      TORCH_FN(manhattan_kernel_impl));
}

} // namespace ops
} // namespace falkon

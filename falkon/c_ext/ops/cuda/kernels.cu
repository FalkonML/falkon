/*
 * copyright PyTorch authors 2026.
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/Exceptions.h>

#include <math.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/macros/Macros.h>

#include "../helpers.h"

namespace falkon {
namespace ops {
namespace {

constexpr int kCUDANumThreads = 256;


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
    scalar_t * result, 
    const scalar_t * x1, 
    const scalar_t * x2,
    const int64_t r2, 
    const int64_t m, 
    const int64_t r_size, 
    const int64_t l1_size, 
    const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const scalar_t * const start = x1 + l * l1_size + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = x2 + l * l2_size + j * m + threadIdx.x;

  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    agg += std::abs(*a - *b);
  }
  __shared__ scalar_t agg_smem[kCUDANumThreads];
  scalar_t agg_init{0.0};
  agg = at::native::cuda_utils::BlockReduce(agg, DistReduceOp<scalar_t>{}, agg_init, agg_smem);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = agg;
  }
}

template <typename scalar_t>
__global__ static void manhattan_kernel_cuda_impl_F(
    scalar_t * result,
    const scalar_t * x1,
    const scalar_t * x2,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size,
    const int64_t r1) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;
  // sizes:
  // x1: [l, r1, m] stride: [r1*m, 1, r1]
  // x2: [l, r2, m] stride: [r2*m, 1, r2]
  // result: [l, r1, r2] stride: [r1*r2, 1, r1]
  // access pattern:
  // x1[l, i, d] = x1[l * l1_size + d * r1 + i]
  // x2[l, j, d] = x2[l * l2_size + d * r2 + j]
  //
  // For a fixed (i,j), consecutive feature dimensions are
  // separated by r1/r2 in memory.

  const scalar_t * a =
      x1 + l * l1_size + i + threadIdx.x * r1;

  const scalar_t * b =
      x2 + l * l2_size + j + threadIdx.x * r2;

  scalar_t agg = 0.0;

  for (int64_t d = threadIdx.x; d < m; d += stride) {
    agg += std::abs(*a - *b);

    a += stride * r1;
    b += stride * r2;
  }

  __shared__ scalar_t agg_smem[kCUDANumThreads];

  scalar_t agg_init{0.0};

  agg = at::native::cuda_utils::BlockReduce(
      agg,
      DistReduceOp<scalar_t>{},
      agg_init,
      agg_smem);

  if (threadIdx.x == 0) {
    // F-contiguous result:
    //
    // result[l, i, j] =
    //     result[l * r1 * r2 + j * r1 + i]
    result[l * r1 * r2 + j * r1 + i] = agg;
  }
}

bool is_fortran_contiguous(const at::Tensor& x) {
    return x.stride(-2) == 1 &&
           x.stride(-1) == x.size(-2);
}

at::Tensor manhattan_kernel_impl(at::Tensor& result, const at::Tensor& x1, const at::Tensor& x2) {
  CHECK_CUDA(result);
  CHECK_CUDA(x1);
  CHECK_CUDA(x2);
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;
  const dim3 grid(result.numel());
  const dim3 block(kCUDANumThreads);

  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cdist_cuda", [&] {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto impl_fptr = manhattan_kernel_cuda_impl_C<scalar_t>;
    if (is_fortran_contiguous(x1)) {
      auto impl_fptr = manhattan_kernel_cuda_impl_F<scalar_t>;
    }
    impl_fptr<<<grid, block, stream>>>(
        result.mutable_data_ptr<scalar_t>(), 
        x1.const_data_ptr<scalar_t>(), 
        x2.const_data_ptr<scalar_t>(),
        r2, 
        m, 
        r_size, 
        l1_size, 
        l2_size
    );
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

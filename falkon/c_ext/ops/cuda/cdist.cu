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
__global__ static void manhattan_kernel_cuda_impl(
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

void manhattan_kernel_impl(at::Tensor& result, const at::Tensor& x1, const at::Tensor& x2) {
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
    auto impl_fptr = cdist_kernel_cuda_impl<scalar_t>;
    impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::manhattan_dist"),
      TORCH_FN(manhattan_kernel_impl));
}

} // namespace ops
} // namespace falkon

#include "../helpers.h"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>

namespace falkon {
namespace ops {
namespace {

template <typename scalar_t, typename acc_t=scalar_t>
struct NormTwoSquareOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + data * data;
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

template <typename scalar_t>
void square_vector_norm_impl(at::TensorIterator& iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
  at::native::gpu_reduce_kernel<scalar_t, scalar_t>(iter, NormTwoSquareOps<scalar_t>(), 0);
}

at::Tensor square_norm_kernel(const at::Tensor& input, int64_t dim, bool keepdim) {
    at::IntArrayRef dimArr = at::IntArrayRef(dim);
    at::ScalarType in_dtype = input.scalar_type();

    // Create the output tensor
    auto result_shape = shape_from_dim(input, dim, keepdim);
    at::Tensor result = at::empty(result_shape, input.options());

    at::TensorIterator iter = at::native::make_reduction("vector_sqnorm", result, input, dimArr, keepdim, in_dtype);
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "square_vector_norm_impl", [&] {
        square_vector_norm_impl<scalar_t>(iter);
    });
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::square_norm"),
      TORCH_FN(square_norm_kernel));
}

} // namespace ops
} // namespace falkon

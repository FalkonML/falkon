#pragma once

#include <type_traits>
#include <c10/macros/Macros.h>

#if defined(__CUDACC__)
#include <THC/THCDeviceUtils.cuh>
#endif

#if defined(__CUDACC__)
void square_vector_norm_cuda_impl(TensorIterator& iter);
#endif

torch::Tensor square_norm(torch::Tensor input, int opt_dim, bool keepdim);

// This accumulator template is used to calculate the order two norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
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

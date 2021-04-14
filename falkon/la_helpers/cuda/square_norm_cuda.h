#pragma once

#include <torch/extension.h>

#include <c10/macros/Macros.h>

torch::Tensor square_norm_cuda(const torch::Tensor input, int dim, torch::optional<bool> opt_keepdim);


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

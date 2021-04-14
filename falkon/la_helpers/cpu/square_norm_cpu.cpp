#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>

#include "square_norm_cpu.h"
#include "../cuda/square_norm_cuda.h"

template <typename scalar_t>
void square_vector_norm_cpu_impl(at::TensorIterator& iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
  at::native::binary_kernel_reduce<scalar_t, scalar_t>(iter, NormTwoSquareOps<scalar_t>(), 0);
}


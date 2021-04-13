#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>

#include "square_norm.h"

template <typename scalar_t>
void square_vector_norm_cuda_impl(TensorIterator& iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
  gpu_reduce_kernel<scalar_t, scalar_t>(iter, NormTwoSquareOps<scalar_t>(), 0);
}

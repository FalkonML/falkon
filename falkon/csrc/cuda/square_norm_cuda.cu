#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDACachingAllocator.h>

#include "square_norm_cuda.h"
#include "../util.h"

template <typename scalar_t>
void square_vector_norm_cuda_impl(at::TensorIterator& iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
  at::native::gpu_reduce_kernel<scalar_t, scalar_t>(iter, NormTwoSquareOps<scalar_t>(), 0);
}

torch::Tensor square_norm_cuda(const torch::Tensor& input, int64_t dim, torch::optional<bool> opt_keepdim) {
    at::IntArrayRef dimArr = at::IntArrayRef(dim);
    at::ScalarType in_dtype = input.scalar_type();
    bool keepdim = opt_keepdim.value_or(false);

    // Create the output tensor
    auto result_shape = shape_from_dim(input, dim, keepdim);
    torch::Tensor result = torch::empty(result_shape, input.options());

    at::TensorIterator iter = at::native::make_reduction("vector_sqnorm", result, input, dimArr, keepdim, in_dtype);
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "square_vector_norm_cuda", [&] {
        square_vector_norm_cuda_impl<scalar_t>(iter);
    });
    return result;
}


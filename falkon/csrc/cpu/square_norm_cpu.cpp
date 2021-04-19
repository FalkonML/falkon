#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>

#include "square_norm_cpu.h"
#include "../util.h"

template <typename scalar_t>
void square_vector_norm_cpu_impl(at::TensorIterator iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
 // at::native::binary_kernel_reduce<NormTwoSquareOpsCPU, scalar_t>(iter, NormTwoSquareOpsCPU<scalar_t>(), (scalar_t)0.0);
  at::native::binary_kernel_reduce(iter, NormTwoSquareOpsCPU<scalar_t>(), (scalar_t)0.0);
}

torch::Tensor square_norm_cpu(const torch::Tensor &input, int64_t dim, torch::optional<bool> opt_keepdim) {
    at::IntArrayRef dimArr = at::IntArrayRef(dim);
    at::ScalarType in_dtype = input.scalar_type();
    bool keepdim = opt_keepdim.value_or(false);
                
    // Create the output tensor
    auto result_shape = shape_from_dim(input, dim, keepdim);
    torch::Tensor result = torch::empty(result_shape, input.options());
    at::TensorIterator iter = at::native::make_reduction("vector_sqnorm", result, input, dimArr, keepdim, in_dtype);
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "square_vector_norm_cpu", [&] {
        square_vector_norm_cpu_impl<scalar_t>(iter);
    });
    return result;
}

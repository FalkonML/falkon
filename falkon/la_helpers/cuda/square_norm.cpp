#include <torch/extension.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/DeviceType.h>

#include "square_norm.h"


template <typename scalar_t>
void square_vector_norm_cpu_impl(TensorIterator& iter) {
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }
  binary_kernel_reduce<scalar_t, scalar_t>(iter, NormTwoSquareOps<scalar_t>(), 0);
}


torch::Tensor square_norm(torch::Tensor input, int opt_dim, bool keepdim) {
    IntArrayRef dim = IntArrayRef{opt_dim};
    ScalarType in_dtype = in.scalar_type();

    // Create the output tensor
    torch::Tensor result = create_reduction_result(inpyt, dim, keepdim, in_dtype);
    ScalarType out_dtype = result.scalar_type();

    auto iter = make_reduction("vector_sqnorm", result, input, dim, keepdim, in_dtype, out_dtype);
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "square_vector_norm_cuda", [&] {
        // TODO: This won't compile on CPU
        if (iter.device_type() == c10::DeviceType::CUDA) {
            square_vector_norm_cuda_impl<scalar_t>(iter);
        } else if(iter.device_type() == c10::DeviceType::CPU) {
            square_vector_norm_cpu_impl<scalar_t>(iter);
        } else {
            TORCH_CHECK("Square norm not implemented for ", iter.device_type(), " device type.;")
        }
    });
}

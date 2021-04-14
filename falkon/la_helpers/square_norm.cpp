#include <torch/extension.h>

#include <ATen/ATen.h>
//#include <ATen/DimVector.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "square_norm.h"

#include "cpu/square_norm_cpu.h"
#ifdef WITH_CUDA
#include "cuda/square_norm_cuda.h"
#endif



inline at::DimVector shape_from_dim(const torch::Tensor& tensor, int dim, bool keepdim) {
    auto shape = at::DimVector(tensor.sizes());
    for (int d = shape.size() - 1; d >= 0; d++) {
        if (d == dim) {
            if (keepdim) {
                shape[d] = 1;
            } else {
                shape.erase(shape.begin() + d);
            }
        }
    }
    return shape;
}

torch::Tensor square_norm(const torch::Tensor input, int opt_dim, bool keepdim) {
    at::IntArrayRef dim = at::IntArrayRef{opt_dim};
    at::ScalarType in_dtype = input.scalar_type();

    // Create the output tensor
    auto result_shape = shape_from_dim(input, opt_dim, keepdim);
    torch::Tensor result = torch::empty(result_shape, input.options());

    auto iter = at::native::make_reduction("vector_sqnorm", result, input, dim, keepdim, in_dtype);
    AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "square_vector_norm_cuda", [&] {
        if (input.device().is_cuda()) {
#ifdef WITH_CUDA
            square_vector_norm_cuda_impl<scalar_t>(iter);
#else
            TORCH_CHECK("Not compiled with CUDA support");
#endif
        } else {
            square_vector_norm_cpu_impl<scalar_t>(iter);
        }
    });
    return result;
}

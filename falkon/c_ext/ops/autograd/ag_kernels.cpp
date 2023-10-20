#include "../kernels.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace falkon {
namespace ops {
namespace {

class RbfKernelFunction
        : public torch::autograd::Function<RbfKernelFunction> {
    public:
        static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                                  const torch::autograd::Variable& m1,
                                  const torch::autograd::Variable& m2,
                                  const torch::autograd::Variable& s)
        {
            at::AutoDispatchBelowADInplaceOrView g;

            at::Tensor kernel = rbf_kernel(m1, m2, s);

            ctx->save_for_backward({m1, m2, s, kernel});
            ctx->saved_data["m1_needs_grad"] = m1.requires_grad();
            ctx->saved_data["m2_needs_grad"] = m2.requires_grad();
            ctx->saved_data["s_needs_grad"] = s.requires_grad();

            return kernel;
        }
        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                     const torch::autograd::tensor_list grad_output)
         {
            auto m1 = ctx->get_saved_variables()[0];
            auto m2 = ctx->get_saved_variables()[1];
            auto s = ctx->get_saved_variables()[2];
            auto kernel = ctx->get_saved_variables()[3];
            auto m1_needs_grad = ctx->saved_data["m1_needs_grad"].toBool();
            auto m2_needs_grad = ctx->saved_data["m2_needs_grad"].toBool();
            auto s_needs_grad = ctx->saved_data["s_needs_grad"].toBool();

            auto grad_out = grad_output[0];
            at::Tensor grad_m1 = ([&]() {
                if (m1_needs_grad) {
                    return at::empty_like(m1, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                } else {
                    return at::Tensor();
                }
            })();
            at::Tensor grad_m2 = ([&]() {
                if (m2_needs_grad) {
                    return at::empty_like(m2, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                } else {
                    return at::Tensor();
                }
            })();
            at::Tensor grad_s = ([&]() {
                if (s_needs_grad) {
                    return at::zeros_like(s, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
                } else {
                    return at::Tensor();
                }
            })();

            rbf_kernel_grad(m1, grad_m1, m2, grad_m2, s, grad_s, kernel, grad_out, m1_needs_grad, m2_needs_grad, s_needs_grad);
            return {
                grad_m1,
                grad_m2,
                grad_s
            };
        }
};

at::Tensor rbf_kernel_autograd(
        const at::Tensor& m1,
        const at::Tensor& m2,
        const at::Tensor& s) {
    return RbfKernelFunction::apply(m1, m2, s);
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel"),
      TORCH_FN(rbf_kernel_autograd));
}

} // namespace ops
} // namespace falkon

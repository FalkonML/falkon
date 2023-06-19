#include "../square_norm.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace falkon {
namespace ops {
namespace {

class SquareNormFunction
        : public torch::autograd::Function<SquareNormFunction> {
    public:
        static torch::autograd::variable_list forward(
                torch::autograd::AutogradContext *ctx,
                const torch::autograd::Variable& input,
                int64_t dim,
                bool keepdim) {
            at::AutoDispatchBelowADInplaceOrView g;
            auto output = square_norm(input, dim, keepdim);

            ctx->save_for_backward({input});
            ctx->saved_data["dim"] = dim;
            ctx->saved_data["keepdim"] = keepdim;

            return {
                output,
            };
        }
        static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext* ctx,
                const torch::autograd::variable_list& grad_output) {
            auto input = ctx->get_saved_variables()[0];

            auto dim = ctx->saved_data["dim"].toInt();
            auto keepdim = ctx->saved_data["keepdim"].toBool();

            auto grad_out = grad_output[0];

            if (!keepdim) {
                grad_out = grad_out.unsqueeze(dim);
            }
            auto grad_input = input * 2;
            grad_input.mul_(grad_out);

            return {
                grad_input,
                torch::autograd::Variable(),
                torch::autograd::Variable()
            };
        }
};

at::Tensor square_norm_autograd(
        const at::Tensor& input,
        int64_t dim,
        bool keepdim) {
    return SquareNormFunction::apply(input, dim, keepdim)[0];
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::square_norm"),
      TORCH_FN(square_norm_autograd));
}

} // namespace ops
} // namespace falkon

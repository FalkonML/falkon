#include <torch/torch.h>

using namespace torch::autograd;


class SquareNormFunction : public Function<SquareNormFunction> {
 public:
    static torch::Tensor forward(
        AutogradContext *ctx, torch::Tensor input, int64_t dim, bool opt_keepdim = false) {
        ctx->save_for_backward({input});
        #ifdef NEW_TORCH
            if (input.device().is_cuda()) {
            #ifdef WITH_CUDA
                return square_norm_cuda(input, dim, opt_keepdim);
            #else
               TORCH_CHECK(false, "Not compiled with CUDA support");
            #endif
            } else {
                return square_norm_cpu(input, dim, opt_keepdim);
            }
        #else
            return at::pow(at::norm(input, 2, dim, opt_keepdim), 2);
        #endif
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto input_grad = input * 2;
    }

    const torch::Tensor &input, int64_t dim, torch::optional<bool> opt_keepdim)

  // bias is an optional argument

  static torch::Tensor forward(
      AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};

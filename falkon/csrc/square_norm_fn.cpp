#include <torch/torch.h>
#include <torch/script.h>

#ifdef NEW_TORCH
    #include "cpu/square_norm_cpu.h"
    #ifdef WITH_CUDA
        #include "cuda/square_norm_cuda.h"
    #endif
#endif

using torch::autograd::variable_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;


torch::Tensor square_norm_fw(torch::Tensor input, int64_t dim, bool keepdim) {
    #ifdef NEW_TORCH
        if (input.device().is_cuda()) {
            #ifdef WITH_CUDA
                return square_norm_cuda(input, dim, keepdim);
            #else
                AT_ERROR("Not compiled with CUDA support");
            #endif
        } else {
            return square_norm_cpu(input, dim, keepdim);
        }
    #else
        return at::pow(at::norm(input, 2, dim, keepdim), 2);
    #endif
}

class SquareNormFunction : public Function<SquareNormFunction> {
 public:
    static variable_list forward(AutogradContext *ctx,
                                 Variable input,
                                 int64_t dim,
                                 torch::optional<bool> opt_keepdim) {
        bool keepdim = false;
        if (opt_keepdim.has_value()) {
            keepdim = opt_keepdim.value();
        }

        ctx->saved_data["dim"] = dim;
        ctx->saved_data["keepdim"] = keepdim;

        ctx->save_for_backward({input});
        auto out = square_norm_fw(input, dim, keepdim);
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
        auto input = ctx->get_saved_variables()[0];
        auto dim = ctx->saved_data["dim"].toInt();
        auto keepdim = ctx->saved_data["keepdim"].toBool();
        auto grad_out = grad_outputs[0];

        if (!keepdim) {
            grad_out = grad_out.unsqueeze(dim);
        }
        auto grad_input = input * 2;
        grad_input.mul_(grad_out);

        return {grad_input, Variable(), Variable()};
    }
};


torch::Tensor scatter_mean(torch::Tensor input, int64_t dim, torch::optional<bool> keepdim) {
  return SquareNormFunction::apply(input, dim, keepdim)[0];
}


static auto registry = torch::RegisterOperators()
                        .op("falkon::square_norm", &scatter_mean);

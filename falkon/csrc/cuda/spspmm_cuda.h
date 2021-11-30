#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spspmm_cuda(
    const torch::Tensor &rowptrA,
    const torch::Tensor &colA,
    const torch::Tensor &valA,
    const torch::Tensor &rowptrB,
    const torch::Tensor &colB,
    const torch::Tensor &valB,
    int64_t N
);

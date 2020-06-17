#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spspmm_cuda(
    torch::Tensor rowptrA, 
    torch::Tensor colA,
    torch::Tensor valA,
    torch::Tensor rowptrB, 
    torch::Tensor colB,
    torch::Tensor valB, 
    int64_t K
);

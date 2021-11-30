#pragma once

#include <torch/extension.h>

torch::Tensor csr2dense_cuda(
    const torch::Tensor &rowptr,
    const torch::Tensor &col,
    const torch::Tensor &val,
    torch::Tensor &out
);

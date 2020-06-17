#pragma once

#include <torch/extension.h>

torch::Tensor csr2dense_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor val,
    torch::Tensor out
);
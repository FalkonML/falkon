#pragma once

#include <torch/extension.h>

torch::Tensor sparse_bdot_impl(
    const torch::Tensor &indexptr1,
    const torch::Tensor &indices1,
    const torch::Tensor &data1,
    const torch::Tensor &indexptr2,
    const torch::Tensor &indices2,
    const torch::Tensor &data2,
    torch::optional<torch::Tensor> out=torch::nullopt);

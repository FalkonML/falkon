#pragma once

#include <torch/extension.h>

torch::Tensor norm(torch::Tensor indexptr, torch::Tensor data,
                   torch::optional<torch::Tensor> out=torch::nullopt);

torch::Tensor norm_sq(torch::Tensor indexptr, torch::Tensor data,
                      torch::optional<torch::Tensor> out=torch::nullopt);
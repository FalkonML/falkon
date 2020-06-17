#pragma once

#include <torch/extension.h>

torch::Tensor addspmm(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor valA,
                      torch::Tensor colptrB, torch::Tensor rowB, torch::Tensor valB,
                      torch::optional<torch::Tensor> out=torch::nullopt);
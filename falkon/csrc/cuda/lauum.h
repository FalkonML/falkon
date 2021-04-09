#pragma once

#include <torch/extension.h>

torch::Tensor lauum_cuda(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb, const bool lower);


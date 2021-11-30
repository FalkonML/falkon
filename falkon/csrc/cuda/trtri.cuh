#pragma once

#include <torch/extension.h>

torch::Tensor trtri_cuda(torch::Tensor &A, const bool lower, const bool unitdiag);

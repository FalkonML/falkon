#pragma once

#include <torch/extension.h>


torch::Tensor square_norm(torch::Tensor input, int opt_dim, bool keepdim);


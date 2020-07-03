#include <torch/extension.h>

torch::Tensor cuda_copy_triang(torch::Tensor &A,
                               bool upper);

torch::Tensor cuda_mul_triang(torch::Tensor &A,
                         bool upper,
                         bool preserve_diag,
                         double multiplier);

#include <torch/extension.h>

//torch::Tensor lauum_lower(torch::Tensor &input, torch::Tensor &output);
torch::Tensor lauum_lower(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb);


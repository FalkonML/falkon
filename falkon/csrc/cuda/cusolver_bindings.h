#pragma once

#include <torch/extension.h>

int cusolver_potrf_buffer_size(const torch::Tensor &A, bool upper, int n, int lda);

void cusolver_potrf(const torch::Tensor& A, const torch::Tensor& workspace, const torch::Tensor& info, int workspace_size, bool upper, int n, int lda);

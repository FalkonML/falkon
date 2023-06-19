#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

int64_t cusolver_potrf_buffer_size(
        at::Tensor &A,
        bool upper,
        int64_t n,
        int64_t lda);
void cusolver_potrf(
        at::Tensor& A,
        at::Tensor& workspace,
        at::Tensor& info,
        int64_t workspace_size,
        bool upper,
        int64_t n,
        int64_t lda);

} // namespace ops
} // namespace falkon

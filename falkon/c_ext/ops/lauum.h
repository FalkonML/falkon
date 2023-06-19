#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor lauum(
        const int64_t n,
        const at::Tensor &A,
        const int64_t lda,
        at::Tensor &B,
        const int64_t ldb,
        const bool lower);

} // namespace ops
} // namespace falkon

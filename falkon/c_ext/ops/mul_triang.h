#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor mul_triang(
        at::Tensor &mat,
        const double multiplier,
        const bool upper,
        const bool preserve_diag);

} // namespace ops
} // namespace falkon

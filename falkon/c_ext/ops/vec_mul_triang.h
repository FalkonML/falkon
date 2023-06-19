#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor vec_mul_triang(
        at::Tensor &mat,
        const at::Tensor &multiplier_vec,
        const bool upper,
        const bool side);

} // namespace ops
} // namespace falkon

#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor square_norm(
    const at::Tensor &self,
    int64_t dim,
    bool keepdim=false);

} // namespace ops
} // namespace falkon

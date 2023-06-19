#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor copy_triang(
    at::Tensor &self,
    const bool upper);

} // namespace ops
} // namespace falkon

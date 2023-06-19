#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor copy_transpose(
    const at::Tensor &self,
    at::Tensor &out);

} // namespace ops
} // namespace falkon

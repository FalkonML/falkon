#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
spspmm(
    const at::Tensor &rowptrA,
    const at::Tensor &colA,
    const at::Tensor &valA,
    const at::Tensor &rowptrB,
    const at::Tensor &colB,
    const at::Tensor &valB,
    int64_t N);

} // namespace ops
} // namespace falkon

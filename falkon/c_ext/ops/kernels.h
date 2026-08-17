#pragma once

#include <ATen/ATen.h>


namespace falkon {
namespace ops {

at::Tensor cdist_l1_out(
    const at::Tensor& x1,
    const at::Tensor& x2,
    at::Tensor& out);

} // namespace ops
} // namespace falkon
#pragma once

#include <ATen/ATen.h>


namespace falkon {
namespace ops {

at::Tensor manhattan_dist(
    at::Tensor &out,
    at::Tensor &x1,
    at::Tensor &x2);

} // namespace ops
} // namespace falkon
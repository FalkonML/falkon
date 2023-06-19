#pragma once

#include <optional>
#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor sparse_bdot(
        const at::Tensor &indexptr1,
        const at::Tensor &indices1,
        const at::Tensor &data1,
        const at::Tensor &indexptr2,
        const at::Tensor &indices2,
        const at::Tensor &data2,
        at::Tensor &out);
at::Tensor sparse_square_norm(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out);
at::Tensor sparse_norm(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out);

} // namespace ops
} // namespace falkon

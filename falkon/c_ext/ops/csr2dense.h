#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor csr2dense(
        const at::Tensor &rowptr,
        const at::Tensor &col,
        const at::Tensor &val,
        at::Tensor &out);

} // namespace ops
} // namespace falkon

#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor potrf(
        at::Tensor &mat,
        bool upper,
        bool clean,
        bool overwrite);

at::Tensor parallel_potrf(
     c10::IntArrayRef devices,
     c10::IntArrayRef block_starts,
     c10::IntArrayRef block_ends,
     c10::IntArrayRef block_sizes,
     c10::IntArrayRef block_devices,
     c10::IntArrayRef block_ids,
     at::Tensor& A);

} // namespace ops
} // namespace falkon

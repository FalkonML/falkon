#include "kernels.h"

#include <torch/library.h>
#include <torch/types.h>
// #include <torch/extension.h>
#include <ATen/native/Distance.h>

namespace falkon {
namespace ops {

at::Tensor cdist_l1_out(
    const at::Tensor& x1,
    const at::Tensor& x2,
    at::Tensor& out)
{
    TORCH_CHECK(x1.dim() >= 2, "x1 must have at least 2 dimensions");
    TORCH_CHECK(x2.dim() >= 2, "x2 must have at least 2 dimensions");

    TORCH_CHECK(
        x1.size(-1) == x2.size(-1),
        "x1 and x2 must have the same feature dimension");

    TORCH_CHECK(
        out.device() == x1.device(),
        "out, x1 and x2 must be on the same device");

    TORCH_CHECK(
        out.scalar_type() == x1.scalar_type() &&
        out.scalar_type() == x2.scalar_type(),
        "x1, x2 and out must have the same dtype");

    const auto M = x1.size(-2);
    const auto N = x2.size(-2);

    TORCH_CHECK(
        out.size(-2) == M && out.size(-1) == N,
        "incorrect output shape");

    // This is the exact PyTorch cdist kernel dispatch.
    at::native::cdist_stub(
        x1.device().type(),
        out,
        x1,
        x2,
        /*p=*/1.0
    );

    return out;
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
//   m.def(TORCH_SELECTIVE_SCHEMA(
//       "falkon::cdist_l1_out(Tensor x1, Tensor x2, Tensor(a!) out) -> Tensor(a!)"));
  m.def("falkon::cdist_l1_out(Tensor x1, Tensor x2, Tensor(a!) out) -> Tensor(a!)");
}

// TORCH_LIBRARY_FRAGMENT(falkon, m) {
//     m.def(
//         "l1_out(Tensor x1, Tensor x2, Tensor(a!) out) -> Tensor(a!)"
//     );
// }

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
    m.impl("l1_out", &cdist_l1_out);
}

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
    m.impl("l1_out", &cdist_l1_out);
}


} // namespace ops
} // namespace falkon

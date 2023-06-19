#include "mul_triang.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor mul_triang(
        at::Tensor &mat,
        const double multiplier,
        const bool upper,
        const bool preserve_diag) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::mul_triang", "")
                       .typed<decltype(mul_triang)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        mat,
        multiplier,
        upper,
        preserve_diag
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::mul_triang(Tensor(a!) mat, float multiplier, bool upper, bool preserve_diag) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

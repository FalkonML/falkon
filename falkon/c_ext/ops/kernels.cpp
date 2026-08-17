#include "kernels.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {
    
at::Tensor manhattan_dist(
    at::Tensor &out,
    at::Tensor &x1,
    at::Tensor &x2) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::manhattan_dist", "")
                       .typed<decltype(manhattan_dist)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    op.call(
        out, x1, x2
    );
    return out;
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::manhattan_dist(Tensor(a!) out, Tensor x1, Tensor x2) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

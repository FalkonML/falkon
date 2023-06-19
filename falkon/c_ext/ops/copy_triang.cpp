#include "copy_triang.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor copy_triang(
        at::Tensor &self,
        const bool upper) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::copy_triang", "")
                       .typed<decltype(copy_triang)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        self,
        upper
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::copy_triang(Tensor(a!) self, bool upper) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

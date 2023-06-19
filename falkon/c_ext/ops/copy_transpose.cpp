#include "copy_transpose.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor copy_transpose(
        const at::Tensor &self,
        at::Tensor &out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::copy_transpose", "")
                       .typed<decltype(copy_transpose)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        self,
        out
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::copy_transpose(Tensor self, Tensor(a!) out) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

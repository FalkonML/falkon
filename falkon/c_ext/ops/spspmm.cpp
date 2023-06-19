#include "spspmm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
spspmm(
        const at::Tensor &rowptrA,
        const at::Tensor &colA,
        const at::Tensor &valA,
        const at::Tensor &rowptrB,
        const at::Tensor &colB,
        const at::Tensor &valB,
        int64_t N) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::spspmm", "")
                       .typed<decltype(spspmm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rowptrA,
        colA,
        valA,
        rowptrB,
        colB,
        valB,
        N
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::spspmm(Tensor rowptrA, Tensor colA, Tensor valA, Tensor rowptrB, Tensor colB, Tensor valB, int N) -> (Tensor, Tensor, Tensor)"));
}

} // namespace ops
} // namespace falkon

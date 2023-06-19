#include "lauum.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor lauum(
        const int64_t n,
        const at::Tensor &A,
        const int64_t lda,
        at::Tensor &B,
        const int64_t ldb,
        const bool lower) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::lauum", "")
                       .typed<decltype(lauum)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        n,
        A,
        lda,
        B,
        ldb,
        lower
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::lauum(int n, Tensor A, int lda, Tensor(a!) B, int ldb, bool lower) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

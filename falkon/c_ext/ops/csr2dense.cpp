#include "csr2dense.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor csr2dense(
        const at::Tensor &rowptr,
        const at::Tensor &col,
        const at::Tensor &val,
        at::Tensor &out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::csr2dense", "")
                       .typed<decltype(csr2dense)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rowptr,
        col,
        val,
        out
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::csr2dense(Tensor rowptr, Tensor col, Tensor val, Tensor(a!) out) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

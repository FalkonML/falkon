#include "sparse_vector_ops.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor sparse_norm(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::sparse_norm", "")
                       .typed<decltype(sparse_norm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        indexptr,
        data,
        out
    );
}
at::Tensor sparse_square_norm(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::sparse_square_norm", "")
                       .typed<decltype(sparse_square_norm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        indexptr,
        data,
        out
    );
}
at::Tensor sparse_bdot(
        const at::Tensor &indexptr1,
        const at::Tensor &indices1,
        const at::Tensor &data1,
        const at::Tensor &indexptr2,
        const at::Tensor &indices2,
        const at::Tensor &data2,
        at::Tensor &out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::sparse_bdot", "")
                       .typed<decltype(sparse_bdot)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        indexptr1,
        indices1,
        data1,
        indexptr2,
        indices2,
        data2,
        out
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::sparse_square_norm(Tensor indexptr, Tensor data, *, Tensor(a!) out) -> Tensor(a!)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::sparse_norm(Tensor indexptr, Tensor data, *, Tensor(a!) out) -> Tensor(a!)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::sparse_bdot(Tensor indexptr1, Tensor indices1, Tensor data1, Tensor indexptr2, Tensor indices2, Tensor data2, *, Tensor (a!) out) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

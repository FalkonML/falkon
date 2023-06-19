#include "vec_mul_triang.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor vec_mul_triang(
        at::Tensor &mat,
        const at::Tensor &multiplier_vec,
        const bool upper,
        const bool side) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::vec_mul_triang", "")
                       .typed<decltype(vec_mul_triang)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        mat,
        multiplier_vec,
        upper,
        side
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::vec_mul_triang(Tensor (a!) mat, Tensor multiplier_vec, bool upper, bool side) -> Tensor (a!)"));
}

} // namespace ops
} // namespace falkon

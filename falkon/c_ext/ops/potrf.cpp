#include "potrf.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor potrf(
        at::Tensor &mat,
        bool upper,
        bool clean,
        bool overwrite) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::potrf", "")
                       .typed<decltype(potrf)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        mat,
        upper,
        clean,
        overwrite
    );
}

at::Tensor parallel_potrf(
     c10::IntArrayRef devices,
     c10::IntArrayRef block_starts,
     c10::IntArrayRef block_ends,
     c10::IntArrayRef block_sizes,
     c10::IntArrayRef block_devices,
     c10::IntArrayRef block_ids,
     at::Tensor& A) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::parallel_potrf", "")
                       .typed<decltype(parallel_potrf)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        devices,
        block_starts,
        block_ends,
        block_sizes,
        block_devices,
        block_ids,
        A
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::potrf(Tensor(a!) mat, bool upper, bool clean, bool overwrite) -> Tensor(a!)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::parallel_potrf(int[] devices, int[] block_starts, int[] block_ends, int[] block_sizes, int[] block_devices, int[] block_ids, Tensor(a!) A) -> Tensor(a!)"));
}

} // namespace ops
} // namespace falkon

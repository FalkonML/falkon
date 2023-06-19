#include "cusolver_bindings.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

int64_t cusolver_potrf_buffer_size(
        at::Tensor &A,
        bool upper,
        int64_t n,
        int64_t lda) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cusolver_potrf_buffer_size", "")
                       .typed<decltype(cusolver_potrf_buffer_size)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        upper,
        n,
        lda
    );
}
void cusolver_potrf(
        at::Tensor& A,
        at::Tensor& workspace,
        at::Tensor& info,
        int64_t workspace_size,
        bool upper,
        int64_t n,
        int64_t lda) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cusolver_potrf", "")
                       .typed<decltype(cusolver_potrf)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        workspace,
        info,
        workspace_size,
        upper,
        n,
        lda
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cusolver_potrf_buffer_size(Tensor(a!) A, bool upper, int n, int lda) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cusolver_potrf(Tensor(a!) A, Tensor(b!) workspace, Tensor(c!) info, int workspace_size, bool upper, int n, int lda) -> ()"));
}

} // namespace ops
} // namespace falkon

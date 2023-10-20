#include "kernels.h"
#include <type_traits>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor rbf_kernel_out(const at::Tensor         & m1,
                          const at::Tensor         & m2,
                          const at::Tensor         & s,
                                at::Tensor         & out) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::rbf_kernel_out", "")
                       .typed<decltype(rbf_kernel_out)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(m1, m2, s, out);
}

at::Tensor rbf_kernel(const at::Tensor         & m1,
                      const at::Tensor         & m2,
                      const at::Tensor         & s) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::rbf_kernel", "")
                       .typed<decltype(rbf_kernel)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(m1, m2, s);
}

void rbf_kernel_grad(const at::Tensor         & m1,
                           at::Tensor         & m1_grad,
                     const at::Tensor         & m2,
                           at::Tensor         & m2_grad,
                     const at::Tensor         & s,
                           at::Tensor         & s_grad,
                     const at::Tensor         & ker,
                     const at::Tensor         &out_g,
                           bool               m1_requires_grad,
                           bool               m2_requires_grad,
                           bool               s_requires_grad) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::rbf_kernel_grad", "")
                       .typed<decltype(rbf_kernel_grad)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    op.call(m1, m1_grad, m2, m2_grad, s, s_grad, ker, out_g, m1_requires_grad, m2_requires_grad, s_requires_grad);
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {

  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::rbf_kernel_out(Tensor m1, Tensor m2, Tensor s, *, Tensor(a!) out) -> (Tensor(a!))"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::rbf_kernel(Tensor m1, Tensor m2, Tensor s) -> (Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::rbf_kernel_grad(Tensor m1, Tensor m1_grad, Tensor m2, Tensor m2_grad, Tensor s, Tensor s_grad, Tensor ker, Tensor out_g, bool m1_requires_grad, bool m2_requires_grad, bool s_requires_grad) -> ()"));
}

} // namespace ops
} // namespace falkon

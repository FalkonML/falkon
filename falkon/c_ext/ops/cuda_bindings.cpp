#include "cuda_bindings.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

void cuda_2d_copy_async(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cuda_2d_copy_async", "")
                       .typed<decltype(cuda_2d_copy_async)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        dest_tensor,
        dest_pitch,
        src_tensor,
        src_pitch,
        width,
        height
    );
}
void cuda_2d_copy(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cuda_2d_copy", "")
                       .typed<decltype(cuda_2d_copy)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        dest_tensor,
        dest_pitch,
        src_tensor,
        src_pitch,
        width,
        height
    );
}
void cuda_1d_copy_async(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cuda_1d_copy_async", "")
                       .typed<decltype(cuda_1d_copy_async)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        dest_tensor,
        src_tensor,
        count
    );
}
void cuda_1d_copy(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cuda_1d_copy", "")
                       .typed<decltype(cuda_1d_copy)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        dest_tensor,
        src_tensor,
        count
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
//  m.def(TORCH_SELECTIVE_SCHEMA(
//      "falkon::mem_get_info(int device_id) -> (int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cuda_2d_copy_async(Tensor (a!) dest_tensor, int dest_pitch, Tensor src_tensor, int src_pitch, int width, int height) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cuda_2d_copy(Tensor (a!) dest_tensor, int dest_pitch, Tensor src_tensor, int src_pitch, int width, int height) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cuda_1d_copy_async(Tensor (a!) dest_tensor, Tensor src_tensor, int count) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cuda_1d_copy(Tensor (a!) dest_tensor, Tensor src_tensor, int count) -> ()"));
}

} // namespace ops
} // namespace falkon

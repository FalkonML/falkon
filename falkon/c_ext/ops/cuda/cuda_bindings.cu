#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../helpers.h"

namespace falkon {
namespace ops {

namespace {

std::tuple<int64_t, int64_t> mem_get_info(int64_t device_id) {
    const at::cuda::CUDAGuard device_guard(device_id);
    size_t free;
    size_t total;
    C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return std::tuple<int64_t, int64_t>{
        (int64_t)free,
        (int64_t)total
    };
}

void cuda_2d_copy_async(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        dest_tensor.data_ptr(),
        dest_pitch,
        src_tensor.data_ptr(),
        src_pitch,
        width,
        height,
        cudaMemcpyDefault,
        torch_stream.stream()
    ));
}

void cuda_2d_copy(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height) {
    C10_CUDA_CHECK(cudaMemcpy2D(
        dest_tensor.data_ptr(),
        dest_pitch,
        src_tensor.data_ptr(),
        src_pitch,
        width,
        height,
        cudaMemcpyDefault
    ));
}

void cuda_1d_copy_async(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dest_tensor.data_ptr(),
        src_tensor.data_ptr(),
        count,
        cudaMemcpyDefault,
        torch_stream.stream()
    ));
}

void cuda_1d_copy(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count) {
    C10_CUDA_CHECK(cudaMemcpy(
        dest_tensor.data_ptr(),
        src_tensor.data_ptr(),
        count,
        cudaMemcpyDefault
    ));
}

} // namespace

// registered as catch-all function since it has no tensor inputs, and dispatcher doesn't know what to do
TORCH_LIBRARY_FRAGMENT(falkon, m) {
    m.def("falkon::mem_get_info", &mem_get_info);
}

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
//  m.impl(
//      TORCH_SELECTIVE_NAME("falkon::mem_get_info"),
//      TORCH_FN(mem_get_info));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cuda_2d_copy_async"),
      TORCH_FN(cuda_2d_copy_async));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cuda_2d_copy"),
      TORCH_FN(cuda_2d_copy));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cuda_1d_copy_async"),
      TORCH_FN(cuda_1d_copy_async));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::cuda_1d_copy"),
      TORCH_FN(cuda_1d_copy));
}

} // ops
} // falkon

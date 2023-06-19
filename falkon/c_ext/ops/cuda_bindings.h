#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {
std::tuple<int64_t, int64_t> mem_get_info(int64_t device_id);
void cuda_2d_copy_async(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height);
void cuda_2d_copy(
        at::Tensor& dest_tensor,
        const int64_t dest_pitch,
        const at::Tensor& src_tensor,
        const int64_t src_pitch,
        const int64_t width,
        const int64_t height);
void cuda_1d_copy_async(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count);
void cuda_1d_copy(
        at::Tensor& dest_tensor,
        const at::Tensor &src_tensor,
        const int64_t count);
} // namespace ops
} // namespace falkon

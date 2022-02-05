#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

std::pair<size_t, size_t> mem_get_info(int device_id);

void cuda_2d_copy_async(
    torch::Tensor& dest_tensor,
    const int dest_pitch,
    const torch::Tensor& src_tensor,
    const int src_pitch,
    const int width,
    const int height,
    const at::cuda::CUDAStream &stream
);

void cuda_2d_copy(
    torch::Tensor& dest_tensor,
    const int dest_pitch,
    const torch::Tensor& src_tensor,
    const int src_pitch,
    const int width,
    const int height
);

void cuda_1d_copy_async(
    torch::Tensor& dest_tensor,
    const torch::Tensor &src_tensor,
    const int count,
    const at::cuda::CUDAStream &stream
);

void cuda_1d_copy(
    torch::Tensor& dest_tensor,
    const torch::Tensor &src_tensor,
    const int count
);

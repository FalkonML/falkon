#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


void cublas_2d_copy_to_dev_async (
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& host_tensor,
    const int lda, torch::Tensor& dev_tensor,
    const int ldb,
    const at::cuda::CUDAStream &stream);

void cublas_2d_copy_to_dev (
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& host_tensor,
    const int lda, torch::Tensor& dev_tensor,
    const int ldb);

void cublas_2d_copy_to_host_async(
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& dev_tensor,
    const int lda, torch::Tensor& host_tensor,
    const int ldb,
    const at::cuda::CUDAStream &stream);

void cublas_2d_copy_to_host(
    const int rows,
    const int cols,
    const int elemSize,
    const torch::Tensor& dev_tensor,
    const int lda, torch::Tensor& host_tensor,
    const int ldb);

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

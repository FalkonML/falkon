#pragma once

#include <torch/extension.h>


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


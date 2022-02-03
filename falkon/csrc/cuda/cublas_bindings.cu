#include "cublas_bindings.h"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cublas_v2.h>
#include "utils.cuh"


void cublas_2d_copy_to_dev_async (const int rows, const int cols, const int elemSize, const torch::Tensor& host_tensor, const int lda, torch::Tensor& dev_tensor, const int ldb, const at::cuda::CUDAStream &stream) {
    TORCH_CUDABLAS_CHECK(cublasSetMatrixAsync(
        rows, cols, elemSize,
        host_tensor.data_ptr(),
        lda,
        dev_tensor.data_ptr(),
        ldb,
        stream.stream()
    ));
}

void cublas_2d_copy_to_dev (const int rows, const int cols, const int elemSize, const torch::Tensor& host_tensor, const int lda, torch::Tensor& dev_tensor, const int ldb) {
    TORCH_CUDABLAS_CHECK(cublasSetMatrix(
        rows, cols, elemSize,
        host_tensor.data_ptr(),
        lda,
        dev_tensor.data_ptr(),
        ldb
    ));
}

void cublas_2d_copy_to_host_async(const int rows, const int cols, const int elemSize, const torch::Tensor& dev_tensor, const int lda, torch::Tensor& host_tensor, const int ldb, const at::cuda::CUDAStream &stream) {
    TORCH_CUDABLAS_CHECK(cublasGetMatrixAsync(
        rows, cols, elemSize,
        dev_tensor.data_ptr(),
        lda,
        host_tensor.data_ptr(),
        ldb,
        stream.stream()
    ));
}

void cublas_2d_copy_to_host(const int rows, const int cols, const int elemSize, const torch::Tensor& dev_tensor, const int lda, torch::Tensor& host_tensor, const int ldb) {
    TORCH_CUDABLAS_CHECK(cublasGetMatrix(
        rows, cols, elemSize,
        dev_tensor.data_ptr(),
        lda,
        host_tensor.data_ptr(),
        ldb
    ));
}


void cuda_2d_copy_async(
    torch::Tensor& dest_tensor,
    const int dest_pitch,
    const torch::Tensor& src_tensor,
    const int src_pitch,
    const int width,
    const int height,
    const at::cuda::CUDAStream &stream
)
{
    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        dest_tensor.data_ptr(),
        dest_pitch,
        src_tensor.data_ptr(),
        src_pitch,
        width,
        height,
        cudaMemcpyDefault,
        stream.stream()
    ));
}

void cuda_2d_copy( 
    torch::Tensor& dest_tensor,
    const int dest_pitch,
    const torch::Tensor& src_tensor,
    const int src_pitch,
    const int width,
    const int height
)
{

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
    torch::Tensor& dest_tensor,
    const torch::Tensor &src_tensor,
    const int count,
    const at::cuda::CUDAStream &stream
)
{
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dest_tensor.data_ptr(),
        src_tensor.data_ptr(),
        count,
        cudaMemcpyDefault,
        stream.stream()
    ));
}

void cuda_1d_copy(
    torch::Tensor& dest_tensor,
    const torch::Tensor &src_tensor,
    const int count
)
{
    C10_CUDA_CHECK(cudaMemcpy(
        dest_tensor.data_ptr(),
        src_tensor.data_ptr(),
        count,
        cudaMemcpyDefault
    ));
}


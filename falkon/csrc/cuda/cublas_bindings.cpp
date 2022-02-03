#include <torch/extension.h>
#include <cublas_v2.h>


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
        ldb,
        stream.stream()
    ));
}

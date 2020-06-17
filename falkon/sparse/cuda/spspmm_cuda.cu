#include "spspmm_cuda.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cusparse.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define DISPATCH_SPSPMM_TYPES(TYPE, ...)                                       \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparseXcsrgemm2_bufferSizeExt =                            \
          cusparseScsrgemm2_bufferSizeExt;                                     \
      const auto &cusparseXcsrgemm2 = cusparseScsrgemm2;                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparseXcsrgemm2_bufferSizeExt =                            \
          cusparseDcsrgemm2_bufferSizeExt;                                     \
      const auto &cusparseXcsrgemm2 = cusparseDcsrgemm2;                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spspmm_cuda(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor valA,
            torch::Tensor rowptrB, torch::Tensor colB, torch::Tensor valB, 
	    int64_t K) {
  /* Input checks: all matrices should be in CSR format, matrix `D` is not used.
   * C = alpha*A*B + beta*D
   * A: m x k
   * B: k x n
   * D: m x n
   * C: m x n
   */
  CHECK_CUDA(rowptrA);
  CHECK_CUDA(colA);
  CHECK_CUDA(valA);
  CHECK_CUDA(rowptrB);
  CHECK_CUDA(colB);
  CHECK_CUDA(valB);

  CHECK_INPUT(rowptrA.dim() == 1);
  CHECK_INPUT(colA.dim() == 1);
  CHECK_INPUT(valA.dim() == 1);
  CHECK_INPUT(valA.size(0) == colA.size(0));

  CHECK_INPUT(rowptrB.dim() == 1);
  CHECK_INPUT(colB.dim() == 1);
  CHECK_INPUT(valB.dim() == 1);
  CHECK_INPUT(valB.size(0) == colB.size(0));

  auto scalar_type = valA.scalar_type();
  /*
  * Summary of the necessary steps
  * 1. Allocate buffer for working-memory of size given by the cusparseXcsrgemm2_bufferSizeExt function
  * 2. Compute the row-pointers of the output C with function cusparseXcsrgemm2Nnz. This calculates the nnzC
  * 4. allocates csrValC and csrColIndC of nnzC elements respectively, and fill them with the
  *    cusparseXcsrgemm2 function
  */
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseStatus_t status;
  cudaError_t cuda_status;
  auto device = rowptrA.get_device();
  c10::cuda::CUDAGuard g(device);

  // Creates default matrix descriptor (0-based and GENERAL matrix)
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  // Convert indices to int (could be long at input)
  rowptrA = rowptrA.toType(torch::kInt);
  colA = colA.toType(torch::kInt);
  rowptrB = rowptrB.toType(torch::kInt);
  colB = colB.toType(torch::kInt);

  int64_t M = rowptrA.numel() - 1;
  int64_t N = rowptrB.numel() - 1;
  auto rowptrA_data = rowptrA.data_ptr<int>();
  auto colA_data = colA.data_ptr<int>();
  auto rowptrB_data = rowptrB.data_ptr<int>();
  auto colB_data = colB.data_ptr<int>();
  int nnzA = colA.numel();
  int nnzB = colB.numel();
  // These values will be computed by this function
  torch::Tensor rowptrC, colC, valC;
  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;

  // Step 1: Create an opaque structure.
  csrgemm2Info_t info = NULL;
  cusparseCreateCsrgemm2Info(&info);

  // Step 2: Allocate buffer for `csrgemm2Nnz` and `csrgemm2`.
  size_t bufferSize;
  DISPATCH_SPSPMM_TYPES(scalar_type, [&] {
    scalar_t alpha = (scalar_t)1.0;
    status = cusparseXcsrgemm2_bufferSizeExt(handle, M, N, K, &alpha,
        descr, colA.numel(), rowptrA_data, colA_data,
        descr, colB.numel(), rowptrB_data, colB_data,
        NULL, descr,  0, NULL, NULL, // Describes matrix D which is not used
        info, &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cusparseDestroyMatDescr(descr);
      cusparseDestroyCsrgemm2Info(info);
      AT_ERROR("cusparse csrgemm2_bufferSizeExt function failed with error code '", status, "'.");
    }

    void *buffer = NULL;
    cuda_status = cudaMalloc(&buffer, bufferSize);
    if (cuda_status != cudaSuccess) {
      cusparseDestroyMatDescr(descr);
      cusparseDestroyCsrgemm2Info(info);
      AT_ERROR("cuda malloc failed with error code '", cuda_status, "'.");
    }

    // Step 3: Compute CSR row pointer. This will fill `rowptrC_data` and `nnzC`
    rowptrC = torch::empty(M + 1, rowptrA.options());
    auto rowptrC_data = rowptrC.data_ptr<int>();
    status = cusparseXcsrgemm2Nnz(handle, M, N, K,
        descr, colA.numel(), rowptrA_data, colA_data,
        descr, colB.numel(), rowptrB_data, colB_data,
        descr, 0, NULL, NULL,
        descr, rowptrC_data, nnzTotalDevHostPtr,
        info, buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cudaFree(buffer);
      cusparseDestroyMatDescr(descr);
      cusparseDestroyCsrgemm2Info(info);
      AT_ERROR("cusparse csrgemm2Nnz function failed with error code '", status, "'.");
    }

    // Step 4: Compute CSR entries.
    colC = torch::empty(nnzC, rowptrC.options());
    auto colC_data = colC.data_ptr<int>();

    valC = torch::empty(nnzC, valA.options());
    auto valC_data = valC.data_ptr<scalar_t>();
    auto valA_data = valA.data_ptr<scalar_t>();
    auto valB_data = valB.data_ptr<scalar_t>();

    status = cusparseXcsrgemm2(handle, M, N, K, &alpha,
        descr, colA.numel(), valA_data, rowptrA_data, colA_data,
        descr, colB.numel(), valB_data, rowptrB_data, colB_data,
        NULL, descr, 0, NULL, NULL, NULL,  // Describes matrix D
        descr, valC_data, rowptrC_data, colC_data,
        info, buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cudaFree(buffer);
      cusparseDestroyMatDescr(descr);
      cusparseDestroyCsrgemm2Info(info);
      AT_ERROR("cusparse csrgemm2 function failed with error code '", status, "'.");
    }
    cudaFree(buffer);
  });

  // Step 5: Free the opaque structure.
  cusparseDestroyCsrgemm2Info(info);
  cusparseDestroyMatDescr(descr);

  return std::make_tuple(rowptrC, colC, valC);
}

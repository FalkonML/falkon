#include "spspmm_cuda.h"

#include <cusparse.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "utils.cuh"


#if defined(__CUDACC__) && (CUSPARSE_VERSION >= 11000)// || (!defined(_MSC_VER) && CUSPARSE_VERSION >= 10301))
#define IS_GENERIC_AVAILABLE() 1
#else
#define IS_GENERIC_AVAILABLE() 0
#endif

#if IS_GENERIC_AVAILABLE()
#include <library_types.h>
#endif

#if IS_GENERIC_AVAILABLE()

template<typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
run_spspmm_cuda(
    const torch::Tensor &rowptrA,
    const torch::Tensor &colA,
    const torch::Tensor &valA,
    const torch::Tensor &rowptrB,
    const torch::Tensor &colB,
    const torch::Tensor &valB,
    int64_t N
)
{
  constexpr auto cuda_value_type = std::is_same<float, scalar_t>::value ? CUDA_R_32F : CUDA_R_64F;
  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSpMatDescr_t matA, matB, matC;
  void*  dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  const int64_t M = rowptrA_int.numel() - 1, K = rowptrB_int.numel() - 1;
  const int nnzA = valA.numel(), nnzB = valB.numel();
  const scalar_t alpha = (scalar_t)1.0, beta = (scalar_t)0.0;
  auto& allocator = c10::cuda::CUDACachingAllocator::get();

  // Convert indices to int (could be long at input)
  const torch::Tensor &rowptrA_int = rowptrA.toType(torch::kInt);
  const torch::Tensor &colA_int = colA.toType(torch::kInt);
  const torch::Tensor &rowptrB_int = rowptrB.toType(torch::kInt);
  const torch::Tensor &colB_int = colB.toType(torch::kInt);

  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(&matA, M, K, nnzA, rowptrA_int.data_ptr<int>(), colA_int.data_ptr<int>(),
                                           valA.data_ptr<scalar_t>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, cuda_value_type));
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(&matB, K, N, nnzB, rowptrB_int.data_ptr<int>(), colB_int.data_ptr<int>(),
                                           valB.data_ptr<scalar_t>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, cuda_value_type));
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(&matC, M, N, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, cuda_value_type));

  // Step 0. Opaque object creation
  cusparseSpGEMMDescr_t spgemmDesc;
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

  // Step 1. Estimate amount of work (buffer 1)
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       &alpha, matA, matB, &beta, matC, cuda_value_type, CUSPARSE_SPGEMM_DEFAULT,
                                                       spgemmDesc, &bufferSize1, NULL));
  at::DataPtr dataPtr1 = allocator.allocate(bufferSize1);
  dBuffer1 = dataPtr1.get();
  // Step 2. Fill buffer 1?
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       &alpha, matA, matB, &beta, matC, cuda_value_type, CUSPARSE_SPGEMM_DEFAULT,
                                                       spgemmDesc, &bufferSize1, dBuffer1));
  // Step 3. Estimate amount of work (buffer 2)
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, matB, &beta, matC, cuda_value_type, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize2, NULL));
  at::DataPtr dataPtr2 = allocator.allocate(bufferSize2);
  dBuffer2 = dataPtr2.get();
  // Step 4. compute the intermediate product of A * B
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, matB, &beta, matC, cuda_value_type, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize2, dBuffer2));
  // Step 5. Retrieve nnzC from matrix descriptor, allocate all of C and update pointers in matC descriptor
  int64_t C_num_rows, C_num_cols, nnzC;
  TORCH_CUDASPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &nnzC));

  torch::Tensor rowptrC = torch::empty(M + 1, rowptrA_int.options());
  torch::Tensor colC = torch::empty(nnzC, rowptrC.options());
  torch::Tensor valC = torch::empty(nnzC, valA.options());

  TORCH_CUDASPARSE_CHECK(cusparseCsrSetPointers(matC, rowptrC.data_ptr<int>(), colC.data_ptr<int>(),
                                                valC.data_ptr<scalar_t>()));
  // Step 6. Copy the final products to matrix C
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha, matA, matB, &beta, matC, cuda_value_type, CUSPARSE_SPGEMM_DEFAULT,
                                             spgemmDesc));

  // Finally free-up temporary buffers
  TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(matA));
  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(matB));
  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(matC));
  return std::make_tuple(rowptrC, colC, valC);
}

#else

template<typename value_t>
cusparseStatus_t cusparseXcsrgemm2_bufferSizeExt(
        cusparseHandle_t         handle,
        int                      m,
        int                      n,
        int                      k,
        const value_t*           alpha,
        const cusparseMatDescr_t descrA,
        int                      nnzA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        const cusparseMatDescr_t descrB,
        int                      nnzB,
        const int*               csrRowPtrB,
        const int*               csrColIndB,
        const value_t*           beta,
        const cusparseMatDescr_t descrD,
        int                      nnzD,
        const int*               csrRowPtrD,
        const int*               csrColIndD,
        csrgemm2Info_t           info,
        size_t*                  pBufferSizeInBytes) { }
template<>
cusparseStatus_t cusparseXcsrgemm2_bufferSizeExt<float>(
        cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA,
        const int* csrColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const float* beta,
        const cusparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
{
    return cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB,
                                           csrRowPtrB, csrColIndB, beta, descrD, nnzD, csrRowPtrD, csrColIndD, info,
                                           pBufferSizeInBytes);
}
template<>
cusparseStatus_t cusparseXcsrgemm2_bufferSizeExt<double>(
        cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA,
        const int* csrColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const double* beta,
        const cusparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
{
    return cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB,
                                           csrRowPtrB, csrColIndB, beta, descrD, nnzD, csrRowPtrD, csrColIndD, info,
                                           pBufferSizeInBytes);
}
template<typename value_t>
cusparseStatus_t cusparseXcsrgemm2(
        cusparseHandle_t         handle,
        int                      m,
        int                      n,
        int                      k,
        const value_t*           alpha,
        const cusparseMatDescr_t descrA,
        int                      nnzA,
        const value_t*           csrValA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        const cusparseMatDescr_t descrB,
        int                      nnzB,
        const value_t*           csrValB,
        const int*               csrRowPtrB,
        const int*               csrColIndB,
        const value_t*           beta,
        const cusparseMatDescr_t descrD,
        int                      nnzD,
        const value_t*           csrValD,
        const int*               csrRowPtrD,
        const int*               csrColIndD,
        const cusparseMatDescr_t descrC,
        value_t*                 csrValC,
        const int*               csrRowPtrC,
        int*                     csrColIndC,
        const csrgemm2Info_t     info,
        void*                    pBuffer) { }
template<>
cusparseStatus_t cusparseXcsrgemm2<float>(
        cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrB, int nnzB,
        const float* csrValB, const int* csrRowPtrB, const int* csrColIndB, const float* beta, const cusparseMatDescr_t descrD,
        int nnzD, const float* csrValD, const int* csrRowPtrD, const int* csrColIndD, const cusparseMatDescr_t descrC, float* csrValC,
        const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer)
{
    return cusparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB,
                             csrColIndB, beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, csrRowPtrC, csrColIndC,
                             info, pBuffer);
}
template<>
cusparseStatus_t cusparseXcsrgemm2<double>(
        cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrB, int nnzB,
        const double* csrValB, const int* csrRowPtrB, const int* csrColIndB, const double* beta, const cusparseMatDescr_t descrD,
        int nnzD, const double* csrValD, const int* csrRowPtrD, const int* csrColIndD, const cusparseMatDescr_t descrC, double* csrValC,
        const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer) {
    return cusparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB,
                             csrColIndB, beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, csrRowPtrC, csrColIndC,
                             info, pBuffer);
}

template<typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
run_spspmm_cuda(
    const torch::Tensor &rowptrA,
    const torch::Tensor &colA,
    const torch::Tensor &valA,
    const torch::Tensor &rowptrB,
    const torch::Tensor &colB,
    const torch::Tensor &valB,
    int64_t N
)
{
  /* Input checks: all matrices should be in CSR format, matrix `D` is not used.
   * C = alpha*A*B + beta*D
   * A: m x k
   * B: k x n
   * D: m x n
   * C: m x n
   */
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  /*
  * Summary of the necessary steps
  * 1. Allocate buffer for working-memory of size given by the cusparseXcsrgemm2_bufferSizeExt function
  * 2. Compute the row-pointers of the output C with function cusparseXcsrgemm2Nnz. This calculates the nnzC
  * 4. allocates csrValC and csrColIndC of nnzC elements respectively, and fill them with the
  *    cusparseXcsrgemm2 function
  */

  // Convert indices to int (could be long at input)
  const torch::Tensor &rowptrA_int = rowptrA.toType(torch::kInt);
  const torch::Tensor &colA_int = colA.toType(torch::kInt);
  const torch::Tensor &rowptrB_int = rowptrB.toType(torch::kInt);
  const torch::Tensor &colB_int = colB.toType(torch::kInt);

  const int64_t M = rowptrA_int.numel() - 1, K = rowptrB_int.numel() - 1;
  const int nnzA = valA.numel();
  const int nnzB = valB.numel();
  const scalar_t alpha = (scalar_t)1.0;

  torch::Tensor rowptrC = torch::empty(M + 1, rowptrA_int.options());
  torch::Tensor colC, valC;
  int nnzC;

  // Creates default matrix descriptor (0-based and GENERAL matrix)
  cusparseMatDescr_t descr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&descr));
  // Pointers (to alpha) are in host memory.
  TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

  // Step 1: Create an opaque structure.
  csrgemm2Info_t info = NULL;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));

  // Step 2: Allocate buffer for `csrgemm2Nnz` and `csrgemm2`.
  size_t bufferSize;
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2_bufferSizeExt<scalar_t>(
    handle,
    M, /* Number of columns in C (output) */
    N, /* Number of rows in C (output) */
    K, /* Number of cols in A, rows in B */
    &alpha, /* Multiplier in front of mat-mul */
    descr, /* Matrix descriptor for A */
    nnzA, /* NNZ for A */
    rowptrA_int.data_ptr<int>(), /* Row-pointer array for A */
    colA_int.data_ptr<int>(), /* Column data array for A */
    descr, /* Matrix descriptor for B */
    nnzB, /* NNZ for B */
    rowptrB_int.data_ptr<int>(), /* Row-pointer array for B */
    colB_int.data_ptr<int>(), /* Column data array for B */
    NULL, /* beta (multiplier summed matrix D) */
    descr, /* Matrix descriptor for D */
    0, /* NNZ for D */
    NULL, /* Row-pointer array for D (unused) */
    NULL, /* Column data array for D */
    info,
    &bufferSize /* Output */
  ));

  auto& allocator = c10::cuda::CUDACachingAllocator::get();
  at::DataPtr bufferDataPtr = allocator.allocate(bufferSize);
  auto csrGemmBuffer = bufferDataPtr.get();

  // Step 3: Compute CSR row pointer. This will fill `rowptrC_data` and `nnzC`
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2Nnz(
    handle,
    M, /* Number of columns in C (output) */
    N, /* Number of rows in C (output) */
    K, /* Number of cols in A, rows in B */
    descr, /* Matrix descriptor for A */
    nnzA, /* NNZ for A */
    rowptrA_int.data_ptr<int>(), /* Row-pointer array for A */
    colA_int.data_ptr<int>(), /* Column data array for A */
    descr, /* Matrix descriptor for B */
    nnzB, /* NNZ for B */
    rowptrB_int.data_ptr<int>(), /* Row-pointer array for B */
    colB_int.data_ptr<int>(), /* Column data array for B */
    descr, /* Matrix descriptor for D */
    0, /* NNZ for D */
    NULL, /* Row-pointer array for D (unused) */
    NULL, /* Column data array for D */
    descr, /* Matrix descriptor for C */
    rowptrC.data_ptr<int>(), /* Output: column data array for C */
    &nnzC, /* Output: number of nnz entries in C */
    info,
    bufferDataPtr.get() /* Additional workspace in GPU memory */
  ));

  // Step 4: Compute CSR entries.
  colC = torch::empty(nnzC, rowptrC.options());
  valC = torch::empty(nnzC, valA.options());

  /* C = alpha * A @ B + beta * D (beta, D are empty) */
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2<scalar_t>(
    handle,
    M, /* Number of columns in C (output) */
    N, /* Number of rows in C (output) */
    K, /* Number of cols in A, rows in B */
    &alpha, /* Multiplier in front of mat-mul */
    descr, /* Matrix descriptor for A */
    nnzA, /* NNZ for A */
    valA.data_ptr<scalar_t>(), /* Value array for A */
    rowptrA_int.data_ptr<int>(), /* Row-pointer array for A */
    colA_int.data_ptr<int>(), /* Column data array for A */
    descr, /* Matrix descriptor for B */
    nnzB, /* NNZ for B */
    valB.data_ptr<scalar_t>(), /* Value array for B */
    rowptrB_int.data_ptr<int>(), /* Row-pointer array for B */
    colB_int.data_ptr<int>(), /* Column data array for B */
    NULL, /* beta (multiplier summed matrix D) */
    descr, /* Matrix descriptor for D */
    0, /* NNZ for D */
    NULL, /* Value array for D */
    NULL, /* Row-pointer array for D (unused) */
    NULL, /* Column data array for D */
    descr, /* Matrix descriptor for C */
    valC.data_ptr<scalar_t>(), /* Value array for C */
    rowptrC.data_ptr<int>(), /* Row-pointer array for C */
    colC.data_ptr<int>(), /* Column data array for C */
    info,
    bufferDataPtr.get() /* Additional workspace in GPU memory */
  ));

  // Step 5: Free the opaque structure.
  cusparseDestroyCsrgemm2Info(info);
  cusparseDestroyMatDescr(descr);

  return std::make_tuple(rowptrC, colC, valC);
}
#endif

template<typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spspmm_cuda(
    const torch::Tensor &rowptrA,
    const torch::Tensor &colA,
    const torch::Tensor &valA,
    const torch::Tensor &rowptrB,
    const torch::Tensor &colB,
    const torch::Tensor &valB,
    int64_t N
)
{
  CHECK_CUDA(rowptrA);
  CHECK_CUDA(colA);
  CHECK_CUDA(valA);
  CHECK_CUDA(rowptrB);
  CHECK_CUDA(colB);
  CHECK_CUDA(valB);

  TORCH_CHECK(rowptrA.dim() == 1);
  TORCH_CHECK(colA.dim() == 1);
  TORCH_CHECK(valA.dim() == 1);
  TORCH_CHECK(valA.size(0) == colA.size(0));

  TORCH_CHECK(rowptrB.dim() == 1);
  TORCH_CHECK(colB.dim() == 1);
  TORCH_CHECK(valB.dim() == 1);
  TORCH_CHECK(valB.size(0) == colB.size(0));

  TORCH_CHECK(valA.dtype() == valB.dtype(), "Expected A, B with equal dtypes but found ",
    valA.dtype(), ", ", valB.dtype());

  auto scalar_type = valA.scalar_type();
  at::DeviceGuard g(rowptrA.device());

  AT_DISPATCH_FLOATING_TYPES(valA.scalar_type(), "csr2dense_cuda", [&] {
    return run_spspmm_cuda<scalar_t>(rowptrA, colA, valA, rowptrB, colB, valB, N);
  });
}

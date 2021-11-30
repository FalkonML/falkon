#include "csr2dense_cuda.h"

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

// Helpers for dispatching int32 and int64 (indices). Only available in newer pytorch versions
#ifndef AT_PRIVATE_CASE_TYPE_USING_HINT
#define AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)        \
  case enum_type: {                                                              \
    using HINT = type;                                                           \
    return __VA_ARGS__();                                                        \
  }
#endif
#ifndef AT_DISPATCH_INDEX_TYPES
#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                            \
  [&] {                                                                     \
    at::ScalarType _it = ::detail::scalar_type(TYPE);                       \
    switch (_it) {                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Int, int32_t, index_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)\
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      \
    }                                                                       \
  }()
#endif


#if IS_GENERIC_AVAILABLE()

template <typename value_t, typename index_t>
void run_csr2dense(
        const torch::Tensor &rowptr, 
        const torch::Tensor &col, 
        const torch::Tensor &val, 
        torch::Tensor &out) {
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  constexpr auto cuda_value_type = std::is_same<float, value_t>::value
                                          ? CUDA_R_32F : CUDA_R_64F;
  constexpr auto cusparse_index_type = std::is_same<int32_t, index_t>::value
                                          ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  const auto dense_order = is_fortran_contig(out)
                                          ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

  // Create sparse and dense matrix descriptors
  cusparseSpMatDescr_t csr_mat;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
    /*output=*/&csr_mat,
    /*rows=*/out.size(0),
    /*cols=*/out.size(1),
    /*nnz=*/val.numel(),
    /*csrRowOffsets=*/const_cast<index_t*>(rowptr.data_ptr<index_t>()),
    /*csrColInd=*/const_cast<index_t*>(col.data_ptr<index_t>()),
    /*csrValues=*/const_cast<value_t*>(val.data_ptr<value_t>()),
    /*csrRowOffsetsType=*/cusparse_index_type,
    /*csrColIndType=*/cusparse_index_type,
    /*idxBase=*/CUSPARSE_INDEX_BASE_ZERO,
    /*valueType=*/cuda_value_type
  ));
  cusparseDnMatDescr_t dn_mat;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    /*output=*/&dn_mat,
    /*rows=*/out.size(0),
    /*cols=*/out.size(1),
    /*ld=*/out.stride(1),
    /*values=*/out.data_ptr<value_t>(),
    /*valueType=*/cuda_value_type,
    /*order=*/dense_order
  ));
  // Check needed buffer size, and allocate it
  size_t buf_size;
  TORCH_CUDASPARSE_CHECK(cusparseSparseToDense_bufferSize(
    handle, csr_mat, dn_mat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &buf_size));
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto conv_buf = allocator.allocate(buf_size);
  // Run sparse->dense
  TORCH_CUDASPARSE_CHECK(cusparseSparseToDense(
    handle, csr_mat, dn_mat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, conv_buf.get()));
  // Cleanup
  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(csr_mat));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(dn_mat));
}

#else // Non-generic implementation (using legacy cuSPARSE API)
template<typename value_t>
cusparseStatus_t cusparseXcsr2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const value_t* csrValA,
                                    const int* csrRowPtrA,
                                    const int* csrColIndA,
                                    value_t* A,
                                    int lda) { }
template<>
cusparseStatus_t cusparseXcsr2dense<float>(
      cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, float* A, int lda) {
    return cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}
template<>
cusparseStatus_t cusparseXcsr2dense<double>(
      cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double* A, int lda) {
    return cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}

template <typename value_t, typename index_t>
void run_csr2dense(
        const torch::Tensor &rowptr, 
        const torch::Tensor &col, 
        const torch::Tensor &val, 
        torch::Tensor &out) {
  TORCH_CHECK(out.stride(0) == 1, "Output matrix is not F-contiguous");

  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // Convert indices to int TODO: This may cause problems if it doesn't fit in int!
  auto rowptr_int = rowptr.toType(torch::kInt);
  auto col_int = col.toType(torch::kInt);

  // Creates default matrix descriptor (0-based and GENERAL matrix)
  cusparseMatDescr_t descr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&descr));
  TORCH_CUDASPARSE_CHECK(cusparseXcsr2dense<value_t>(
    handle,                      /* cuSparse handle */
    (int)out.size(0),            /* Number of rows */
    (int)out.size(1),            /* Number of columns */
    descr,                       /* Descriptor for the dense matrix */
    val.data_ptr<value_t>(),     /* Non-zero elements of sparse matrix */
    rowptr_int.data_ptr<int>(),  /* CSR row indices */
    col_int.data_ptr<int>(),     /* CSR column indices */
    out.data_ptr<value_t>(),     /* Output data */
    (int)out.stride(1)           /* Leading dimension of dense matrix */
  ));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(descr));
}
#endif


torch::Tensor csr2dense_cuda(
    const torch::Tensor &rowptr,
    const torch::Tensor &col,
    const torch::Tensor &val,
    torch::Tensor &out) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(val);
  CHECK_CUDA(out);

  const int M = out.size(0);

  TORCH_CHECK(rowptr.numel() - 1 == M, "Expected output with ", rowptr.numel() - 1, " rows but found ", M);
  TORCH_CHECK(val.dtype() == out.dtype(), "Expected csr and output matrix with the same dtypes but found ",
    val.dtype(), " and ", out.dtype());
  TORCH_CHECK(rowptr.device() == col.device() && col.device() == val.device(),
    "Expected all arrays of CSR matrix to be on the same device.");
  TORCH_CHECK(out.device() == val.device(),
    "Expected CSR and dense matrices to be on the same device.");

  at::DeviceGuard g(rowptr.device());
  AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "csr2dense_cuda_value", [&] {
    AT_DISPATCH_INDEX_TYPES(col.scalar_type(), "csr2dense_cuda_index", [&] {
      run_csr2dense<scalar_t, index_t>(rowptr, col, val, out);
    });
  });
  return out;
}

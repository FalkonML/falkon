#include "csr2dense_cuda.h"

#include <cusparse.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "utils.cuh"


#if defined(__CUDACC__) && (CUSPARSE_VERSION >= 11000 || (!defined(_MSC_VER) && CUSPARSE_VERSION >= 10301))
#define IS_GENERIC_AVAILABLE() 1
#else
#define IS_GENERIC_AVAILABLE() 0
#endif

#if IS_GENERIC_AVAILABLE()
#include <library_types.h>
#endif

#if IS_GENERIC_AVAILABLE()
template <class scalar_t>
void run_csr2dense(
        torch::Tensor &rowptr,
        torch::Tensor &col,
        torch::Tensor &val,
        torch::Tensor &out,
        ) {
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  constexpr auto cusparse_value_type = std::is_same<float, scalar_t>::value
                                          ? CUDA_R_32F : CUDA_R_64F;
  const auto cusparse_ind_type = torch::ScalarType::Int == rowptr.scalar_type()
                                          ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  const auto dense_order = is_fortran_contig(out)
                                          ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

  // Create sparse and dense matrix descriptors
  cusparseSpMatDescr_t csr_mat;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
    /*output=*/&csr_mat,
    /*rows=*/rowptr.numel() - 1,
    /*cols=*/out.size(1),
    /*nnz=*/val.numel(),
    /*csrRowOffsets=*/rowptr.data_ptr<scalar_t>(),
    /*csrColInd=*/col.data_ptr<scalar_t>(),
    /*csrValues=*/val.data_ptr<scalar_t>(),
    /*csrRowOffsetsType=*/cusparse_ind_type,
    /*csrColIndType=*/cusparse_ind_type,
    /*idxBase=*/CUSPARSE_INDEX_BASE_ZERO,
    /*valueType=*/cusparse_value_type,
  ));
  cusparseDnMatDescr_t dn_mat;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    /*output=*/&dn_mat,
    /*rows=*/out.size(0),
    /*cols=*/out.size(1),
    /*ld=*/out.stride(1),
    /*values=*/out.data_ptr<scalar_t>(),
    /*valueType=*/cusparse_value_type,
    /*order=*/dense_order,
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

#else
template <class scalar_t>
void run_csr2dense(
        torch::Tensor &rowptr,
        torch::Tensor &col,
        torch::Tensor &val,
        torch::Tensor &out,
        ) {
  TORCH_CHECK(out.stride(0) == 1, "Output matrix is not F-contiguous");

  constexpr auto &cusparseXcsr2dense = std::is_same<float, scalar_t>::value
                                           ? cusparseScsr2dense : cusparseDcsr2dense;
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // Convert indices to int TODO: This may cause problems if it doesn't fit in int!
  rowptr = rowptr.toType(torch::kInt);
  col = col.toType(torch::kInt);

  // Creates default matrix descriptor (0-based and GENERAL matrix)
  cusparseMatDescr_t descr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&descr));
  TORCH_CUDASPARSE_CHECK(cusparseXcsr2dense(
      handle,                     /* cuSparse handle */
      out.size(0),                /* Number of rows */
      out.size(1),                /* Number of columns */
      descr,                      /* Descriptor for the dense matrix */
      val.data_ptr<scalar_t>(),   /* Non-zero elements of sparse matrix */
      rowptr.data_ptr<int>(),     /* CSR row indices */
      col.data_ptr<int>(),        /* CSR column indices */
      out.data_ptr<scalar_t>(),   /* Output data */
      out.stride(1),              /* Leading dimension of dense matrix */
  ));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(descr));
}
#endif


torch::Tensor csr2dense_cuda(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &val, torch::Tensor &out) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(val);
  CHECK_CUDA(out);

  const int M = out.size(0);

  TORCH_CHECK(rowptr.numel() - 1 == M, "Expected output with ", rowptr.numel() - 1, " rows but found ", M);
  TORCH_CHECK_TYPE(val.dtype() == out.dtype(), "Expected csr and output matrix with the same dtypes but found ",
    val.dtype(), " and ", out.dtype());
  TORCH_CHECK(rowptr.device() == col.device() && col.device() == val.device(),
              "Expected all arrays of CSR matrix to be on the same device.");
  TORCH_CHECK(out.device() == val.device(),
              "Expected CSR and dense matrices to be on the same device.");

  at::DeviceGuard g(rowptr.device());
  AT_DISPATCH_FLOATING_TYPES(val.scalar_type(), "csr2dense_cuda", [&] {
    run_csr2dense<scalar_t>(rowptr, col, val, out);
  });
  return out;
}

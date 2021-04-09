#include "csr2dense_cuda.h"
#include "utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>
#include <torch/extension.h>

#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define DISPATCH_CSR2DENSE_TYPES(TYPE, ...)                                    \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparseXcsr2dense = cusparseScsr2dense;                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparseXcsr2dense = cusparseDcsr2dense;                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()


torch::Tensor csr2dense_cuda(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &val, torch::Tensor &out) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(val);
  CHECK_CUDA(out);

  int M = out.size(0);
  int N = out.size(1);
  int ldA = out.stride(1);
  auto scalar_type = val.scalar_type();

  AT_ASSERTM(rowptr.numel() - 1 == M, "Output size does not match sparse tensor size.");
  AT_ASSERTM(val.dtype() == out.dtype(), "Input and output data-types do not match");
  AT_ASSERTM(out.stride(0) == 1, "Output matrix is not F-contiguous");

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseStatus_t status;
  at::DeviceGuard g(rowptr.device());

  // Convert indices to int
  rowptr = rowptr.toType(torch::kInt);
  col = col.toType(torch::kInt);

  // Creates default matrix descriptor (0-based and GENERAL matrix)
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);

  // Extract data pointers for the sparse matrix indices
  auto rowptr_data = rowptr.data_ptr<int>();
  auto col_data = col.data_ptr<int>();

  DISPATCH_CSR2DENSE_TYPES( scalar_type, [&] {
    // Extract data pointers for the dense matrix indices
    auto val_data = val.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    status = cusparseXcsr2dense(handle, M, N,
        descr, val_data, rowptr_data, col_data, out_data, ldA);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cusparseDestroyMatDescr(descr);
      AT_ERROR("cusparse csr2dense function failed with error code '", status, "'.");
    }
  });
  cusparseDestroyMatDescr(descr);
  return out;
}

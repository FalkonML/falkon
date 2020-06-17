#include <torch/extension.h>

#include "cpp/sparse_matmul.h"
#include "cpp/sparse_norm.h"

#ifdef WITH_CUDA
#include "cuda/spspmm_cuda.cuh"
#include "cuda/csr2dense_cuda.cuh"
#endif


torch::Tensor csr2dense(
        torch::Tensor rowptr,
        torch::Tensor col,
        torch::Tensor val,
        torch::Tensor out
    ) {
    if (out.device().is_cuda()) {
#ifdef WITH_CUDA
        return csr2dense_cuda(rowptr, col, val, out);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        AT_ERROR("csr2dense only implemented for CUDA sparse tensors");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> spspmm(
        torch::Tensor rowptrA, torch::Tensor colA,
        torch::Tensor valA,
        torch::Tensor rowptrB, torch::Tensor colB,
        torch::Tensor valB, int64_t K
    ) {
    if (rowptrA.device().is_cuda()) {
#ifdef WITH_CUDA
        return spspmm_cuda(rowptrA, colA, valA, rowptrB, colB, valB, K);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        AT_ERROR("spspmm only implemented for CUDA sparse tensors");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addspmm", &addspmm, "Sparse*Sparse -> Dense matrix multiplication"); // TODO: This may be deleted?
  m.def("norm_sq", &norm_sq, "Squared row-wise norm of a sparse CPU matrix", py::call_guard<py::gil_scoped_release>());
  m.def("spspmm", &spspmm, "Sparse*Sparse -> Sparse matrix multiplication (CUDA tensors)", py::call_guard<py::gil_scoped_release>());
  m.def("csr2dense", &csr2dense, "Convert CSR matrix to dense matrix (CUDA tensors)", py::call_guard<py::gil_scoped_release>());
}

#include <torch/extension.h>
#include <torch/script.h>

#if (TORCH_VERSION_MAJOR >= 1) && (TORCH_VERSION_MINOR >= 7)
#define NEW_TORCH
#endif

#ifdef NEW_TORCH
#include "cpu/square_norm_cpu.h"
#ifdef WITH_CUDA
#include "cuda/square_norm_cuda.h"
#include "cuda/utils.cuh"
#endif
#endif

torch::Tensor square_norm_call(const torch::Tensor &input, int64_t dim, torch::optional<bool> opt_keepdim) {
#ifdef NEW_TORCH
    if (input.device().is_cuda()) {
#ifdef WITH_CUDA
        return square_norm_cuda(input, dim, opt_keepdim);
#else
       TORCH_CHECK(false, "Not compiled with CUDA support");
#endif
    } else {
        return square_norm_cpu(input, dim, opt_keepdim);
    }
#else
    return at::pow(at::norm(input, 2, dim, opt_keepdim.value_or(false)), 2);
#endif
}

torch::Tensor copy_triang_call(torch::Tensor &A, bool upper) {
#ifdef WITH_CUDA
    return cuda_copy_triang(A, upper);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
}

torch::Tensor mul_triang_call(torch::Tensor &A, bool upper, bool preserve_diag, double multiplier) {
#ifdef WITH_CUDA
    return cuda_mul_triang(A, upper, preserve_diag, multiplier);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
}
torch::Tensor transpose_call(torch::Tensor &input, torch::Tensor &output) {
#ifdef WITH_CUDA
    return cuda_transpose(input, output);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
}

torch::Tensor vec_mul_triang_call(torch::Tensor &A, torch::Tensor &v, bool upper, int side) {
#ifdef WITH_CUDA
    return cuda_vec_mul_triang(A, v, upper, side);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
}


TORCH_LIBRARY(my_ops, m) {
  m.def("square_norm", square_norm_call);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_copy_triang", &copy_triang_call, "Make a CUDA tensor symmetric",
        py::arg("A"), py::arg("upper"));

  m.def("cuda_mul_triang", &mul_triang_call, "Multiply the triangular of a CUDA tensor",
        py::arg("A"), py::arg("upper"), py::arg("preserve_diag"), py::arg("multiplier"));

  m.def("cuda_transpose", &transpose_call, "Transpose a matrix out-of-place",
        py::arg("input"), py::arg("output"));

  m.def("cuda_vec_mul_triang", &vec_mul_triang_call, "Multiply a triangular matrix by a vector",
        py::arg("A"), py::arg("v"), py::arg("upper"), py::arg("side"));

  m.def("square_norm", &square_norm_call, "Squared norm", 
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));
}

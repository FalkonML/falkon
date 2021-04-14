#include <torch/extension.h>

#include "cuda/utils.cuh"

#if (TORCH_VERSION_MAJOR >= 1) && (TORCH_VERSION_MINOR >= 7)
#define NEW_TORCH
#endif

#ifdef NEW_TORCH
#include "cpu/square_norm_cpu.h"
#ifdef WITH_CUDA
#include "cuda/square_norm_cuda.h"
#endif
#endif

torch::Tensor square_norm_call(torch::Tensor input, int dim, torch::optional<bool> opt_keepdim) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_copy_triang", &cuda_copy_triang, "Make a CUDA tensor symmetric",
        py::arg("A"), py::arg("upper"));

  m.def("cuda_mul_triang", &cuda_mul_triang, "Multiply the triangular of a CUDA tensor",
        py::arg("A"), py::arg("upper"), py::arg("preserve_diag"), py::arg("multiplier"));

  m.def("cuda_transpose", &cuda_transpose, "Transpose a matrix out-of-place",
        py::arg("input"), py::arg("output"));

  m.def("cuda_vec_mul_triang", &cuda_vec_mul_triang, "Multiply a triangular matrix by a vector",
        py::arg("A"), py::arg("v"), py::arg("upper"), py::arg("side"));

  m.def("square_norm", &square_norm_call, "Squared norm", 
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));
}

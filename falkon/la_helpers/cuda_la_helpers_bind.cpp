#include <torch/extension.h>

#include "cuda/utils.cuh"
#include "square_norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_copy_triang", &cuda_copy_triang, "Make a CUDA tensor symmetric",
        py::arg("A"), py::arg("upper"));

  m.def("cuda_mul_triang", &cuda_mul_triang, "Multiply the triangular of a CUDA tensor",
        py::arg("A"), py::arg("upper"), py::arg("preserve_diag"), py::arg("multiplier"));

  m.def("cuda_transpose", &cuda_transpose, "Transpose a matrix out-of-place",
        py::arg("input"), py::arg("output"));

  m.def("cuda_vec_mul_triang", &cuda_vec_mul_triang, "Multiply a triangular matrix by a vector",
        py::arg("A"), py::arg("v"), py::arg("upper"), py::arg("side"));

  m.def("square_norm", &square_norm, "Squared norm", 
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));
}

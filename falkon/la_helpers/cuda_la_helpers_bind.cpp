#include <vector>
#include <tuple>

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cusolverDn.h>

#include "cuda/utils.cuh"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_copy_triang", &cuda_copy_triang, "Make a CUDA tensor symmetric",
        py::arg("A"), py::arg("upper"));
  m.def("cuda_mul_triang", &cuda_mul_triang, "Multiply the triangular of a CUDA tensor",
        py::arg("A"), py::arg("upper"), py::arg("preserve_diag"), py::arg("multiplier"));
}

#include <vector>
#include <tuple>

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cusolverDn.h>

#include "cuda/multigpu_potrf.cuh"
#include "cuda/lauum.cuh"

static void* ctypes_void_ptr(const py::object& object) {
    PyObject *p_ptr = object.ptr();
    if (!PyObject_HasAttr(p_ptr, PyUnicode_FromString("value"))) {
        return nullptr;
    }
    PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
    if (ptr_as_int == Py_None) {
        return nullptr;
       }
    void *ptr = PyLong_AsVoidPtr(ptr_as_int);
    return ptr;
}


torch::Tensor parallel_potrf(
     std::vector<std::tuple<float, py::object, int>> gpu_info,
     std::vector<std::tuple<int, int, int, int, int>> allocations,
     torch::Tensor A) {
    std::vector<blockAlloc> out_allocs;
    for (std::tuple<int, int, int, int, int> ba_tpl : allocations) {
        blockAlloc ba = {
            .start =std::get<0>(ba_tpl),
            .end   =std::get<1>(ba_tpl),
            .size  =std::get<2>(ba_tpl),
            .device=std::get<3>(ba_tpl),
            .id    =std::get<4>(ba_tpl)
        };
        out_allocs.push_back(ba);
    }
    auto ctypes = py::module::import("ctypes");
    std::vector<gpuInfo> out_gpu_info;
    for (auto &gi_tp : gpu_info) {
        // Parse the cusolver handle
        py::object cus_handle_obj = std::get<1>(gi_tp);
        void *cus_handle_vptr = ctypes_void_ptr(cus_handle_obj);
        if (cus_handle_vptr == nullptr) {
            throw std::invalid_argument("cusolver_handle");
        }
        gpuInfo gi = {
            .free_memory = std::get<0>(gi_tp),
            .cusolver_handle = (cusolverDnHandle_t)cus_handle_vptr,
            .id = std::get<2>(gi_tp)
        };
        out_gpu_info.push_back(gi);
    }
    return parallel_potrf_cuda(out_gpu_info, out_allocs, A);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parallel_potrf", &parallel_potrf, "GPU-Parallel Cholesky Factorization");

  m.def("cuda_lauum_lower", &lauum_lower, "Compute lower-LAUUM",
        py::arg("n"), py::arg("A"), py::arg("lda"), py::arg("B"), py::arg("ldb"));
}

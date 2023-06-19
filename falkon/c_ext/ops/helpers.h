#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace falkon {
namespace ops {

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor.")

inline at::DimVector shape_from_dim(const at::Tensor& tensor, int64_t dim, bool keepdim) {
    auto shape = at::DimVector(tensor.sizes());
    if (dim < 0) {
        // e.g. 3-d tensor, dim is -1 => dim will be 2
        dim = shape.size() + dim;
    }
    for (int64_t d = shape.size() - 1; d >= 0; d--) {
        if (d == dim) {
            if (keepdim) {
                shape[d] = 1;
            } else {
                shape.erase(shape.begin() + d);
            }
        }
    }
    return shape;
}

inline bool is_fortran_contig(const at::Tensor &matrix) {
    return matrix.stride(0) == 1;
}

inline int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}

} // namespace ops
} // namespace falkon

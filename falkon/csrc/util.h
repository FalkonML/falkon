#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>


inline at::DimVector shape_from_dim(const torch::Tensor& tensor, int64_t dim, bool keepdim) {
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

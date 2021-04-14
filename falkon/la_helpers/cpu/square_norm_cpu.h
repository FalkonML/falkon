#pragma once

#include <torch/extension.h>
#include <c10/macros/Macros.h>

torch::Tensor square_norm_cpu(const torch::Tensor input, int dim, torch::optional<bool> opt_keepdim);

template <typename acc_t>
struct NormTwoSquareOpsCPU {
    inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data, int64_t /*idx*/) const {
        return acc + data * data;
    }

    inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
        return a + b;
    }

    inline C10_DEVICE acc_t project(acc_t a) const {
        return a;
    }

    static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
        return acc;
    }
};


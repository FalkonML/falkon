#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")


inline bool is_fortran_contig(const torch::Tensor &matrix) {
    return matrix.stride(0) == 1;
}

inline __device__ int2 tri_index_lower(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        linear_index - row * (row + 1) / 2,
        row
    );
}

inline __device__ int2 tri_index_upper(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        row,
        linear_index - row * (row + 1) / 2
    );
}

inline int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}





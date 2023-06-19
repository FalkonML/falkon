#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/Parallel.h>

namespace falkon {
namespace ops {
namespace {

// TODO: Parallelize
template <typename scalar_t>
void mul_upper_diag(
        scalar_t *data,
        const size_t size,
        const scalar_t mul,
        const int row_stride,
        const int col_stride,
        const bool preserve_diag) {
    const int diagonal_offset = preserve_diag ? 1 : 0;
    for (int i = 0; i < size; i++) {
        for (int j = i + diagonal_offset; j < size; j++) {
            data[i * row_stride + j * col_stride] *= mul;
        }
    }
}

template <typename scalar_t>
void mul_lower_diag(
        scalar_t *data,
        const size_t size,
        const scalar_t mul,
        const int row_stride,
        const int col_stride,
        const bool preserve_diag) {
    const int diagonal_offset = preserve_diag ? -1 : 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j <= (i + diagonal_offset); j++) {
            data[i * row_stride + j * col_stride] *= mul;
        }
    }
}


at::Tensor mul_triang_kernel(
        at::Tensor &mat,
        const double multiplier,
        const bool upper,
        const bool preserve_diag) {
    AT_ASSERTM(mat.dim() == 2, "mat must be 2D");
    const int64_t n = mat.size(0);
    const int64_t m = mat.size(1);
    TORCH_CHECK(
        (n == m),
        "Input matrix must be square. Found shape: (",
        n,
        ", ",
        m,
        ")");
    int64_t row_stride = mat.stride(0);
    int64_t col_stride = mat.stride(1);
    TORCH_CHECK(
        (row_stride == 1 || col_stride == 1),
        "Input must be contiguous in one dimension. Found strides: (",
        row_stride,
        ", ",
        col_stride,
        ")");

    bool bupper = upper;
    if (row_stride == 1) {
        bupper = !upper;
        int tmp_stride = row_stride;
        row_stride = col_stride;
        col_stride = tmp_stride;
    }

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "mul_triang", [&] {
        const scalar_t mul = (scalar_t)multiplier;
        if (bupper) {
            mul_upper_diag(
                mat.data_ptr<scalar_t>(),
                n,
                mul,
                row_stride,
                col_stride,
                preserve_diag);
        } else {
            mul_lower_diag(
                mat.data_ptr<scalar_t>(),
                n,
                mul,
                row_stride,
                col_stride,
                preserve_diag);
        }
    });
    return mat;
}


} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::mul_triang"),
      TORCH_FN(mul_triang_kernel));
}

} // namespace ops
} // namespace falkon

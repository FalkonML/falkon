#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/BatchLinearAlgebra.h>

#include "../helpers.h"
#include "../mul_triang.h"

namespace falkon {
namespace ops {
namespace {

at::Tensor potrf_kernel(
        at::Tensor &mat,
        bool upper,
        bool clean,
        bool overwrite) {
    AT_ASSERTM(mat.dim() == 2, "Input matrix must be 2D");
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

    char uplo;
    if (is_fortran_contig(mat)) {
        uplo = upper ? 'U' : 'L';
    } else {
        uplo = upper ? 'L' : 'U';
    }

    // Copy array if necessary
    if (!overwrite) {
        mat = mat.clone();
    }

    int info = 0;

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "copy_triang", [&] {
        at::native::lapackCholesky<scalar_t>(
            uplo,
            n,
            mat.data_ptr<scalar_t>(),
            row_stride == 1 ? col_stride : row_stride,
            &info);
        TORCH_CHECK(
            (info == 0),
            "LAPACK potrf failed with status ",
            info,
            ". Params: uplo ",
            uplo,
            ", rows ",
            n);
        // Clean non-factorized part of the matrix
    });
    if (clean) {
        falkon::ops::mul_triang(mat, 0.0, !upper, true);
    }
    return mat;
}


} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::potrf"),
      TORCH_FN(potrf_kernel));
}

} // namespace ops
} // namespace falkon

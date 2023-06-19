#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/Parallel.h>

namespace falkon {
namespace ops {
namespace {

/* https://github.com/pytorch/pytorch/blob/4081e924a8d701d5201db3e4b4ed2da60b072d30/aten/src/ATen/native/TriangularOps.cpp */
template <typename scalar_t>
void vec_mul_triang_impl(
        scalar_t* mat,
        const scalar_t* multiplier_vector,
        const int64_t n,
        const int64_t row_stride,
        const int64_t col_stride,
        const bool side,
        const bool upper) {
   /*
    * Multiply triangular matrix by a column or row vector (depending on side, following broadcasting rules)
    * if side == true: multiplier is a row vector
    * if side == false: multiplier is a column vector
    */
    if (col_stride == 1) {
        // C-contiguous (rows are stored as contiguous blocks, stride is (?, 1))
        at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {  // rows
            scalar_t mul;
            for (int64_t i : c10::irange(start, end)) {
                if (!side) {
                    mul = multiplier_vector[i];
                }
                if (upper) {
                    for (int64_t j = i; j < n; j++) {  // cols
                        if (side) {
                            mul = multiplier_vector[j];
                        }
                        mat[i * row_stride + j] = mat[i * row_stride + j] * mul;
                    }
                } else {
                    for (int64_t j = 0; j <= i; j++) { // cols
                        if (side) {
                            mul = multiplier_vector[j];
                        }
                        mat[i * row_stride + j] = mat[i * row_stride + j] * mul;
                    }
                }
            }
        });
    } else {
        // F-contiguous (columns are stored as contiguous blocks, stride is (1, ?))
        at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {  // columns
            scalar_t mul;
            for (int64_t i : c10::irange(start, end)) {
                if (side) {
                    mul = multiplier_vector[i];
                }
                if (upper) {
                    for (int64_t j = 0; j <= i; j++) { // rows
                        if (!side) {
                            mul = multiplier_vector[j];
                        }
                        mat[j + i * col_stride] = mat[j + i * col_stride] * mul;
                    }
                } else {
                    for (int64_t j = i; j < n; j++) { // rows
                        if (!side) {
                            mul = multiplier_vector[j];
                        }
                        mat[j + i * col_stride] = mat[j + i * col_stride] * mul;
                    }
                }
            }
        });
    }
}

at::Tensor vec_mul_triang_kernel(
        at::Tensor &mat,
        const at::Tensor &multiplier_vec,
        const bool upper,
        const bool side) {
    AT_ASSERTM(mat.dim() == 2, "mat must be 2D");
    const int64_t n = mat.size(0);
    const int64_t m = mat.size(1);
    const int64_t k = multiplier_vec.size(0);
    TORCH_CHECK(
        (n == m),
        "Input matrix must be square. Found shape: (",
        n,
        ", ",
        m,
        ")");
    TORCH_CHECK(
        (n == k),
        "Multiplier vector must have the same first dimension as the input matrix. Expected shape: (",
        n,
        ", ) found shape: (",
        m,
        ", )");

    at::Tensor multiplier_vec_c = multiplier_vec.contiguous();
    const int64_t row_stride = mat.stride(0);
    const int64_t col_stride = mat.stride(1);
    TORCH_CHECK(
        (row_stride == 1 || col_stride == 1),
        "Input must be contiguous in one dimension. Found strides: (",
        row_stride,
        ", ",
        col_stride,
        ")");

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "vec_mul_triang", [&] {
        vec_mul_triang_impl<scalar_t>(
            mat.data_ptr<scalar_t>(),
            multiplier_vec_c.data_ptr<scalar_t>(),
            n,
            row_stride,
            col_stride,
            side,
            upper
        );
    });
    return mat;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::vec_mul_triang"),
      TORCH_FN(vec_mul_triang_kernel));
}

} // namespace ops
} // namespace falkon

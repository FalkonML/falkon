#include <algorithm>
#include <cstdint>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/Parallel.h>

#include "../helpers.h"

namespace falkon {
namespace ops {
namespace {

template <class scalar_t>
void copy_triang_impl(scalar_t *mat, const int n, const int stride1, const int stride2, const bool upper) {
    // assume input is f-contiguous (contiguous columns, stride1 == 1)
    assert(stride1 == 1);
    constexpr int64_t TILE_SIZE = 64;   // tune for your L1/L2 cache size
    const int64_t B = std::min<int64_t>(n, TILE_SIZE);
    if (upper) {
        at::parallel_for(0, n, B, [&](int64_t jj_start, int64_t jj_end) {
            for (int64_t jj = jj_start; jj < jj_end; jj += B) {
                int64_t j_end = std::min(jj + B, static_cast<int64_t>(n));
                // The i‑loop must start at j+1, so the earliest possible i is jj+1.
                // Process all i blocks that intersect (j_end, n).
                for (int64_t ii = jj + 1; ii < n; ii += B) {
                    int64_t i_end = std::min(ii + B, static_cast<int64_t>(n));
                    for (int64_t j = jj; j < j_end; ++j) {
                        int64_t i_start = std::max(ii, j + 1);
                        // The block might contain no work if i_start >= i_end.
                        for (int64_t i = i_start; i < i_end; ++i) {
                            mat[i + j * stride2] = mat[j + i * stride2];
                        }
                    }
                }
            }
        });
    } else {
        at::parallel_for(0, n, B, [&](int64_t jj_start, int64_t jj_end) {
            for (int64_t jj = jj_start; jj < jj_end; jj += B) {
                int64_t j_end = std::min(jj + B, static_cast<int64_t>(n));
                // The i‑loop goes up to j, so only i blocks that overlap 0..j-1 matter.
                for (int64_t ii = 0; ii < j_end; ii += B) {
                    int64_t i_end = std::min(ii + B, static_cast<int64_t>(n));
                    for (int64_t j = jj; j < j_end; ++j) {
                        int64_t i_stop = std::min(i_end, j);   // i < j
                        // i_start = ii, but also ensure i_start < i_stop.
                        for (int64_t i = ii; i < i_stop; ++i) {
                            mat[i + j * stride2] = mat[j + i * stride2];
                        }
                    }
                }
            }
        });
    }
    /*
    if (upper) {
        at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
            for (int64_t i : c10::irange(start, end)) {
                for (int64_t j = 0; j < i; j++) {
                    // mat[i, j] = mat[j, i]
                    mat[i * stride1 + j * stride2] = mat[j * stride1 + i * stride2];
                }
            }
        });
    } else {
        at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
            for (int64_t i : c10::irange(start, end)) {
                for (int64_t j = i + 1; j < n; j++) {
                    // mat[i, j] = mat[j, i]
                    mat[i * stride1 + j * stride2] = mat[j * stride1 + i * stride2];
                }
            }
        });
    }
    */
}

at::Tensor copy_triang_kernel(
        at::Tensor &mat,
        const bool upper) {
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

    bool bupper = upper;
    if (!is_fortran_contig(mat)) {
        bupper = !upper;
        int64_t tmp = row_stride;
        row_stride = col_stride;
        col_stride = tmp;
    }
    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "copy_triang", [&] {
        copy_triang_impl<scalar_t>(
            mat.data_ptr<scalar_t>(),
            n,
            row_stride,
            col_stride,
            bupper
        );
    });
    return mat;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::copy_triang"),
      TORCH_FN(copy_triang_kernel));
}

} // namespace ops
} // namespace falkon

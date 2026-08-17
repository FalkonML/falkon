/*
 * copyright PyTorch authors 2026.
 */
#include <algorithm>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>

#include <c10/util/irange.h>

namespace falkon {
namespace ops {
namespace {

template<typename scalar_t>
static void run_parallel_manhattan_C(at::Tensor& result, const at::Tensor& t1, const at::Tensor& t2) {
    const scalar_t * const t1_start = t1.const_data_ptr<scalar_t>();
    const scalar_t * const t2_start = t2.const_data_ptr<scalar_t>();
    int64_t d = t1.size(0);
    int64_t r1 = t1.size(-2);
    int64_t r2 = t2.size(-2);
    int64_t m = t1.size(-1);

    scalar_t * const res_start = result.data_ptr<scalar_t>();
    int64_t combs = r1 * r2;
    int64_t size1 = r1 * m;
    int64_t size2 = r2 * m;

    at::parallel_for(0, combs * d, at::internal::GRAIN_SIZE / (16 * m), [=](int64_t start, int64_t end) {
        scalar_t * res = res_start + start;
        const scalar_t * const res_end = res_start + end;
        int64_t l = start / combs;
        int64_t k = start % combs;
        int64_t i = k / r2;
        int64_t j = k % r2;
        i = i * m;
        j = j * m;

        while (res != res_end) {
            const scalar_t * self_i = t1_start + size1 * l + i;
            const scalar_t * self_j = t2_start + size2 * l + j;

            scalar_t agg = 0;
            for (const auto x : c10::irange(m)) {
                scalar_t a = *(self_i + x);
                scalar_t b = *(self_j + x);

                agg = agg + std::abs(a-b);
            }
            *res = agg;

            res += 1;
            j += m;
            if (j == size2) {
                j = 0;
                i += m;
                if (i == size1) {
                    i = 0;
                    l += 1;
                }
            }
        }
    });
}


template <typename scalar_t>
static void run_parallel_manhattan_F(
    at::Tensor& result,
    const at::Tensor& t1,
    const at::Tensor& t2) {
    // sizes:
    // x1: [d, r1, m] stride: [r1*m, 1, r1]
    // x2: [d, r2, m] stride: [r2*m, 1, r2]
    // result: [d, r1, r2] stride: [r1*r2, 1, r1]
    // access pattern:
    // x1[l, j, k] = x1[l * r1 * m + r1 * k + j]
    // x2[l, j, k] = x2[l * r2 * m + r2 * k + j]

    const scalar_t* const t1_start = t1.const_data_ptr<scalar_t>();
    const scalar_t* const t2_start = t2.const_data_ptr<scalar_t>();
    scalar_t* const res_start = result.data_ptr<scalar_t>();
    const int64_t d  = t1.size(0);
    const int64_t r1 = t1.size(-2);
    const int64_t r2 = t2.size(-2);
    const int64_t m  = t1.size(-1);
    const int64_t size1 = r1 * m;
    const int64_t size2 = r2 * m;
    constexpr int64_t TILE_I = 16;
    constexpr int64_t TILE_J = 16;
    constexpr int64_t TILE_D = 16;

    const int64_t num_i_tiles = (r1 + TILE_I - 1) / TILE_I;
    const int64_t num_j_tiles = (r2 + TILE_J - 1) / TILE_J;
    const int64_t num_tiles = d * num_i_tiles * num_j_tiles;
     // Each parallel task computes one TILE_I x TILE_J output tile.
    at::parallel_for(0, num_tiles, at::internal::GRAIN_SIZE / (TILE_I * TILE_J), [=](int64_t start, int64_t end) {
        for (int64_t tile = start; tile < end; ++tile) {
            int64_t tmp = tile;
            const int64_t tile_ij = num_i_tiles * num_j_tiles;
            const int64_t l = tmp / tile_ij;
            tmp %= tile_ij;

            const int64_t tile_i = tmp / num_j_tiles;
            const int64_t tile_j = tmp % num_j_tiles;
            const int64_t i0 = tile_i * TILE_I;
            const int64_t j0 = tile_j * TILE_J;
            const int64_t i_end = std::min(i0 + TILE_I, r1);
            const int64_t j_end = std::min(j0 + TILE_J, r2);
            const int64_t ni = i_end - i0;
            const int64_t nj = j_end - j0;

            scalar_t accum[TILE_I][TILE_J];
            for (int64_t ii = 0; ii < ni; ++ii) {
                for (int64_t jj = 0; jj < nj; ++jj) {
                    accum[ii][jj] = scalar_t(0);
                }
            }
            const scalar_t* const x1 = t1_start + l * size1 + i0;
            const scalar_t* const x2 = t2_start + l * size2 + j0;

            for (int64_t d0 = 0; d0 < m; d0 += TILE_D) {
                const int64_t d_end = std::min(d0 + TILE_D, m);
                for (int64_t x = d0; x < d_end; ++x) {
                    const scalar_t* const a = x1 + x * r1;
                    const scalar_t* const b = x2 + x * r2;
                    /*
                    * i is contiguous in a.
                    * j is contiguous in b.
                    */
                    for (int64_t ii = 0; ii < ni; ++ii) {
                        const scalar_t av = a[ii];
                        for (int64_t jj = 0; jj < nj; ++jj) {
                            accum[ii][jj] += std::abs(av - b[jj]);
                        }
                    }
                }
            }

            scalar_t* const out = res_start + l * r1 * r2;
            for (int64_t jj = 0; jj < nj; ++jj) {
                scalar_t* const out_j = out + (j0 + jj) * r1 + i0;
                for (int64_t ii = 0; ii < ni; ++ii) {
                    out_j[ii] = accum[ii][jj];
                }
            }
        }
    });
}


bool is_fortran_contiguous(const at::Tensor& x) {
    return x.stride(-2) == 1 && x.stride(-1) == x.size(-2);
}

at::Tensor manhattan_dist_kernel(at::Tensor& result, const at::Tensor& x1, const at::Tensor& x2) {
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cpu_manhattan", [&] {
        if (is_fortran_contiguous(x1)) {
            run_parallel_manhattan_F<scalar_t>(result, x1, x2);
        } else {
            run_parallel_manhattan_C<scalar_t>(result, x1, x2);
        }
    });
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::manhattan_dist"),
      TORCH_FN(manhattan_dist_kernel));
}

} // namespace ops
} // namespace falkon

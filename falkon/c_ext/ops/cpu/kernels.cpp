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

template<typename scalar_t>
static void run_parallel_manhattan_F(
    at::Tensor& result,
    const at::Tensor& t1,
    const at::Tensor& t2) {

    const scalar_t* const t1_start =
        t1.const_data_ptr<scalar_t>();

    const scalar_t* const t2_start =
        t2.const_data_ptr<scalar_t>();

    const int64_t d  = t1.size(0);
    const int64_t r1 = t1.size(-2);
    const int64_t r2 = t2.size(-2);
    const int64_t m  = t1.size(-1);

    scalar_t* const res_start =
        result.data_ptr<scalar_t>();

    const int64_t combs = r1 * r2;
    const int64_t size1 = r1 * m;
    const int64_t size2 = r2 * m;

    at::parallel_for(0, combs * d, at::internal::GRAIN_SIZE / (16 * m), [=](int64_t start, int64_t end) {
        scalar_t* res = res_start + start;
        const scalar_t* const res_end = res_start + end;

        int64_t l = start / combs;
        int64_t k = start % combs;
        int64_t i = k % r1;
        int64_t j = k / r1;

        while (res != res_end) {
            const scalar_t* self_i =
                t1_start + l * size1 + i;

            const scalar_t* self_j =
                t2_start + l * size2 + j;

            scalar_t agg = 0;

            for (const auto x : c10::irange(m)) {
                const scalar_t a =
                    self_i[x * r1];

                const scalar_t b =
                    self_j[x * r2];

                agg += std::abs(a - b);
            }
            *res = agg;

            // F-contiguous result: i is the fastest dimension.
            ++i;
            if (i == r1) {
                i = 0;
                ++j;
                if (j == r2) {
                    j = 0;
                    ++l;
                }
            }
            ++res;
        }
    });
}


bool is_fortran_contiguous(const at::Tensor& x) {
    return x.stride(-2) == 1 &&
           x.stride(-1) == x.size(-2);
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

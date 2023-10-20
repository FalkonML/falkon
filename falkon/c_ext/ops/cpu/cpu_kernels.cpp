#include <math.h>            /* exp */
#include <mutex>
#include <cmath>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/AtomicAddFloat.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/CPUBlas.h>
//#include <c10/util/irange.h>
#include "../square_norm.h"

namespace falkon {
namespace ops {
namespace {

template <typename scalar_t>
void rbfk_k(const at::Tensor &m1,
            const at::Tensor &m2,
            const at::Tensor &s,
            at::Tensor &out)
{
    const auto m1_divs = m1.div(s);
    const auto m2_divs = m2.div(s);
    const auto m1_norm = falkon::ops::square_norm(m1_divs, /*dim=*/1, /*keepdim=*/true);
    const auto m2_norm = falkon::ops::square_norm(m2_divs, /*dim=*/1, /*keepdim=*/true);

    // out contains m1 norm to sum it with matmul result
    out.copy_(m1_norm.expand_as(out));

    using opmath_t = at::opmath_type<scalar_t>;
    // for some reason gemm doesn't compile.
    at::addmm_out(out, out, m1_divs, m2_divs.transpose(-2, -1), opmath_t(1), opmath_t(-2));
//    at::native::cpublas::gemm(
//        at::native::TransposeType::NoTranspose,
//        at::native::TransposeType::Transpose,
//        m1_sizes[0], m2_sizes[0], m1_sizes[1],
//        opmath_t(-2),
//        m1.data_ptr<scalar_t>(), lda,
//        m2.data_ptr<scalar_t>(), ldb,
//        opmath_t(1),
//        out.data_ptr<scalar_t>(), ldc);
    out.add_(m2_norm.transpose(-2, -1).expand_as(out));
    out.clamp_min_(opmath_t(1e-20));
    out.mul_(opmath_t(-0.5));
    out.exp_();
}

template <typename scalar_t>
void rbfk_grad_k(const at::Tensor &m1,
                           at::Tensor &m1_grad,
                     const at::Tensor &m2,
                           at::Tensor &m2_grad,
                     const at::Tensor &s,
                           at::Tensor &s_grad,
                     const at::Tensor &ker,
                     const at::Tensor &out_g,
                     bool             m1_requires_grad,
                     bool             m2_requires_grad,
                     bool             s_requires_grad)
{
    int64_t n = ker.size(0);
    int64_t m = ker.size(1);
    int64_t d = m1.size(1);

    // reduction for gradient wrt sigma
    int64_t ker_s1 = ker.stride(0);
    int64_t ker_s2 = ker.stride(1);  // assume 1?
    int64_t outg_s1 = out_g.stride(0);
    int64_t outg_s2 = out_g.stride(1);  // assume 1?
    int64_t m1_s1 = m1.stride(0);
    int64_t m2_s1 = m2.stride(0);
    int64_t m1g_s1 = 0;
    int64_t m2g_s1 = 0;
    if (m1_requires_grad)
        m1g_s1 = m1_grad.stride(0);
    if (m2_requires_grad)
        m2g_s1 = m2_grad.stride(0);

    scalar_t * C10_RESTRICT m1_ptr      = m1.data_ptr<scalar_t>();
    scalar_t * C10_RESTRICT m2_ptr      = m2.data_ptr<scalar_t>();
    scalar_t * C10_RESTRICT s_ptr       = s.data_ptr<scalar_t>();
    scalar_t * C10_RESTRICT ker_ptr     = ker.data_ptr<scalar_t>();
    scalar_t * C10_RESTRICT outg_ptr    = out_g.data_ptr<scalar_t>();
    scalar_t * C10_RESTRICT m1grad_ptr  = NULL;
    scalar_t * C10_RESTRICT m2grad_ptr  = NULL;
    scalar_t * C10_RESTRICT sgrad_ptr   = NULL;
    if (m1_requires_grad)
        m1grad_ptr = m1_grad.data_ptr<scalar_t>();
    if (m2_requires_grad)
        m2grad_ptr = m2_grad.data_ptr<scalar_t>();
    if (s_requires_grad)
        sgrad_ptr  = s_grad.data_ptr<scalar_t>();

    std::mutex sgrad_mutex;

    constexpr int64_t J_BLOCK_SIZE = 2;
    constexpr int64_t GRAIN_SIZE = 20;
    if (m1_requires_grad) {
        at::parallel_for(0, n, GRAIN_SIZE, [&](int64_t start, int64_t end) {
            c10::SmallVector<scalar_t> m1grad_tmp(d, 0);
            c10::SmallVector<scalar_t> sgrad_tmp(d, 0);
            c10::SmallVector<scalar_t, J_BLOCK_SIZE> kerval_tmp(J_BLOCK_SIZE, 0);
            scalar_t *ker_ptr_t = ker_ptr + start * ker_s1;
            scalar_t *outg_ptr_t = outg_ptr + start * outg_s1;
            scalar_t *m1_ptr_t = m1_ptr + start * m1_s1;
            /*
             *  m1-grad[i, k] = sum_j=1^m  ker(i, j) * [m1(i, k) - m2(j, k)]
             *                = sum_j=1^m  ker(i, j) * m1(i, k) - ker(i, j) * m2(j, k)
             *                = m1(i, k) * sum_j=1^m ker(i, j) - sum_j=1^m ker(i, j) * m2(j, k)
             *                =
             *  - for every i, need to iterate over j -> inconvenient.
             */
            for (int64_t i : c10::irange(start, end)) {
                // reset temporary m1-grad to 0
                for (int64_t k = 0; k < d; k++) {
                    m1grad_tmp[k] = 0;
                }
                scalar_t *m2_ptr_t = m2_ptr;

                for (int64_t j = 0; j < m; j += J_BLOCK_SIZE) {
                    int64_t jj_end = (j + J_BLOCK_SIZE) > m ? m - j : J_BLOCK_SIZE;
                    // load 4 ker vals
                    for (int64_t jj = 0; jj < jj_end; jj++) {
                        kerval_tmp[jj] = ker_ptr_t[(j + jj) * ker_s2];
                    }
                    for (int64_t jj = 0; jj < jj_end; jj++) {
                        kerval_tmp[jj] = kerval_tmp[jj] * outg_ptr_t[(j + jj) * outg_s2];
                    }
                    // process 4 js
                    for (int64_t jj = 0; jj < jj_end; jj++) {
                        int64_t k = 0;
                        for (; k <= d - 2; k += 2) {
                            scalar_t m1_d1 = m1_ptr_t[k];
                            scalar_t m1_d2 = m1_ptr_t[k + 1];
                            scalar_t m2_d1 = m2_ptr_t[k];
                            scalar_t m2_d2 = m2_ptr_t[k + 1];

                            scalar_t ij_dist_k_d1 = m1_d1 - m2_d1;
                            scalar_t ij_dist_k_d2 = m1_d2 - m2_d2;
                            scalar_t tmp_mul1 = ij_dist_k_d1 * kerval_tmp[jj];
                            scalar_t tmp_mul2 = ij_dist_k_d2 * kerval_tmp[jj];
                            m1grad_tmp[k] += tmp_mul1;
                            m1grad_tmp[k + 1] += tmp_mul2;

                            if (s_requires_grad) {
                                sgrad_tmp[k] += ij_dist_k_d1 * tmp_mul1;
                                sgrad_tmp[k + 1] += ij_dist_k_d2 * tmp_mul2;
                            }
                        }
                        for (; k < d; k++) {
                            scalar_t ij_dist_k = m1_ptr_t[k] - m2_ptr_t[k];
                            scalar_t tmp_mul = ij_dist_k * kerval_tmp[jj];
                            m1grad_tmp[k] += tmp_mul;
                            if (s_requires_grad) {
                                sgrad_tmp[k] += ij_dist_k * tmp_mul;
                            }
                        }
                        m2_ptr_t += m2_s1;
                    }
                }
                for (int64_t k = 0; k < d; k++) {
                    m1grad_ptr[i * m1g_s1 + k] = - m1grad_tmp[k] / (s_ptr[k] * s_ptr[k]);
                }
                ker_ptr_t += ker_s1;
                outg_ptr_t += outg_s1;
                m1_ptr_t += m1_s1;
            }
            if (s_requires_grad) {
                // update shared-memory containing s-grad
                const std::lock_guard<std::mutex> lock(sgrad_mutex);
                for (int64_t k = 0; k < d; k++) {
                    sgrad_ptr[k] += sgrad_tmp[k];
                }
            }
        });
        s_requires_grad = false;
    }
    if (m2_requires_grad || s_requires_grad) {
        at::parallel_for(0, m, GRAIN_SIZE, [&](int64_t start, int64_t end) {
            c10::SmallVector<scalar_t> m2grad_tmp = c10::SmallVector<scalar_t>(d, 0);
            c10::SmallVector<scalar_t> sgrad_tmp;
            c10::SmallVector<scalar_t, J_BLOCK_SIZE> kerval_tmp(J_BLOCK_SIZE, 0);
//            if (m2_requires_grad)
//                m2grad_tmp = c10::SmallVector<scalar_t>(d, 0);
            if (s_requires_grad)
                sgrad_tmp = c10::SmallVector<scalar_t>(d, 0);

            scalar_t *ker_ptr_t = ker_ptr + start * ker_s2;
            scalar_t *outg_ptr_t = outg_ptr + start * outg_s2;
            scalar_t *m2_ptr_t = m2_ptr + start * m2_s1;

            for (int64_t j : c10::irange(start, end)) {
                // reset temporary m1-grad to 0
                if (m2_requires_grad) {
                    for (int64_t k = 0; k < d; k++) {
                        m2grad_tmp[k] = 0;
                    }
                }
                scalar_t *m1_ptr_t = m1_ptr;
                for (int64_t i = 0; i < n; i += J_BLOCK_SIZE) {
                    int64_t ii_end = (i + J_BLOCK_SIZE) > n ? n - i : J_BLOCK_SIZE;
                    // load 4 ker vals
                    for (int64_t ii = 0; ii < ii_end; ii++) {
                        kerval_tmp[ii] = ker_ptr_t[(i + ii) * ker_s1 ];
                    }
                    for (int64_t ii = 0; ii < ii_end; ii++) {
                        kerval_tmp[ii] = kerval_tmp[ii] * outg_ptr_t[(i + ii) * outg_s1];
                    }
                    // process 4 'i's
                    for (int64_t ii = 0; ii < ii_end; ii++) {
                        int64_t k = 0;
                        for (; k <= d - 2; k += 2) {
                            scalar_t m1_d1 = m1_ptr_t[k];
                            scalar_t m1_d2 = m1_ptr_t[k + 1];
                            scalar_t m2_d1 = m2_ptr_t[k];
                            scalar_t m2_d2 = m2_ptr_t[k + 1];

                            scalar_t ij_dist_k_d1 = m2_d1 - m1_d1;
                            scalar_t ij_dist_k_d2 = m2_d2 - m1_d2;
                            scalar_t tmp_mul1 = ij_dist_k_d1 * kerval_tmp[ii];
                            scalar_t tmp_mul2 = ij_dist_k_d2 * kerval_tmp[ii];
                            if (m2_requires_grad) {
                                m2grad_tmp[k] += tmp_mul1;
                                m2grad_tmp[k + 1] += tmp_mul2;
                            }
                            if (s_requires_grad) {
                                sgrad_tmp[k] += ij_dist_k_d1 * tmp_mul1;
                                sgrad_tmp[k + 1] += ij_dist_k_d2 * tmp_mul2;
                            }
                        }
                        for (; k < d; k++) {
                            scalar_t ij_dist_k = m2_ptr_t[k] - m1_ptr_t[k];
                            scalar_t tmp_mul = ij_dist_k * kerval_tmp[ii];
                            if (m2_requires_grad) {
                                m2grad_tmp[k] += tmp_mul;
                            }
                            if (s_requires_grad) {
                                sgrad_tmp[k] += ij_dist_k * tmp_mul;
                            }
                        }
                        m1_ptr_t += m1_s1;
                    }
                }
                if (m2_requires_grad) {
                    for (int64_t k = 0; k < d; k++) {
                        m2grad_ptr[j * m2g_s1 + k] = - m2grad_tmp[k] / (s_ptr[k] * s_ptr[k]);
                    }
                }
                ker_ptr_t += ker_s2;
                outg_ptr_t += outg_s2;
                m2_ptr_t += m2_s1;
            }
            if (s_requires_grad) {
                // update shared-memory containing s-grad
                const std::lock_guard<std::mutex> lock(sgrad_mutex);
                for (int64_t k = 0; k < d; k++) {
                    sgrad_ptr[k] += sgrad_tmp[k];
                }
            }
        });
    }
}

at::Tensor rbf_kernel_out_clr(const at::Tensor     & m1,
                          const at::Tensor         & m2,
                          const at::Tensor         & s,
                                at::Tensor         & out)
{
    AT_DISPATCH_FLOATING_TYPES(m1.scalar_type(), "rbf_kernel", [&] {
        rbfk_k<scalar_t>(m1, m2, s, out);
    });
    return out;
}

at::Tensor rbf_kernel_clr(const at::Tensor         & m1,
                          const at::Tensor         & m2,
                          const at::Tensor         & s)
{
    at::Tensor out = at::empty({m1.size(0), m2.size(0)}, m1.options());
    AT_DISPATCH_FLOATING_TYPES(m1.scalar_type(), "rbf_kernel", [&] {
        rbfk_k<scalar_t>(m1, m2, s, out);
    });
    return out;
}

void rbf_kernel_grad_clr(const at::Tensor &m1,
                               at::Tensor &m1_grad,
                         const at::Tensor &m2,
                               at::Tensor &m2_grad,
                         const at::Tensor &s,
                               at::Tensor &s_grad,
                         const at::Tensor &ker,
                         const at::Tensor &out_g,
                         bool             m1_requires_grad,
                         bool             m2_requires_grad,
                         bool             s_requires_grad) {
    AT_DISPATCH_FLOATING_TYPES(m1.scalar_type(), "rbf_kernel_grad", [&] {
        rbfk_grad_k<scalar_t>(m1, m1_grad, m2, m2_grad, s, s_grad, ker, out_g, m1_requires_grad, m2_requires_grad, s_requires_grad);
    });
}
} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel_out"),
      TORCH_FN(rbf_kernel_out_clr));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel"),
      TORCH_FN(rbf_kernel_clr));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel_grad"),
      TORCH_FN(rbf_kernel_grad_clr));
}

} // namespace ops
} // namespace falkon

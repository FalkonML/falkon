#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>

#include <cooperative_groups.h>

#include "../helpers.h"
#include "../square_norm.h"

namespace cg = cooperative_groups;
using namespace at::cuda::detail;
using at::TensorBase;

namespace falkon {
namespace ops {
namespace {

#define THREAD_BLOCK_DIM 256

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(THREAD_BLOCK_DIM)
__global__ void rbfk_backward_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> m1,
      TensorInfo<scalar_t, index_t> m1_grad,
      TensorInfo<scalar_t, index_t> m2,
      TensorInfo<scalar_t, index_t> m2_grad,
      TensorInfo<scalar_t, index_t> s,
      TensorInfo<scalar_t, index_t> s_grad,
      TensorInfo<scalar_t, index_t> ker,
      TensorInfo<scalar_t, index_t> out_grad,
      bool             m1_requires_grad,
      bool             m2_requires_grad,
      bool             s_requires_grad) {
    index_t N = ker.sizes[0];
    index_t M = ker.sizes[1];
    index_t D = m1.sizes[1];
    index_t m1_sN = m1.strides[0];
    index_t m2_sM = m2.strides[0];
    index_t ker_sN = ker.strides[0];
    index_t ker_sM = ker.strides[1];
    index_t outg_sN = out_grad.strides[0];
    index_t outg_sM = out_grad.strides[1];
    index_t m1g_sN = 0;
    if (m1_requires_grad) {
        m1g_sN = m1_grad.strides[0];
    }

    constexpr index_t WARPS_IN_THREAD_BLOCK = THREAD_BLOCK_DIM / 32;
    __shared__ scalar_t shared_kernel[WARPS_IN_THREAD_BLOCK][32];
    __shared__ scalar_t shared_outg[WARPS_IN_THREAD_BLOCK][32];

    cg::thread_block tblock = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tblock);

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t k = index % ((D + 31) & -32);  // round D to higher multiple of 32
        const index_t i = index / D;

        if (k <= D) {
            cg::coalesced_group active_warp = cg::coalesced_threads();
            unsigned active_size = active_warp.size();
            scalar_t *m2_ptr = m2.data + k;
            scalar_t *ker_ptr = ker.data + i * ker_sN;
            scalar_t *outg_ptr = out_grad.data + i * outg_sN;

            scalar_t m1_val = m1.data[i * m1_sN + k];  // coalesced global load
            scalar_t m1_g_acc = 0;
            for (index_t j = 0; j < M; j += active_size,
                                       ker_ptr += active_size * ker_sM,
                                       outg_ptr += active_size * outg_sM,
                                       m2_ptr += active_size * m2_sM) {
                index_t jj_end = (j + active_size) > M ? M - j : active_size;

                // Load kernel values into shared memory (coalesce global memory access)
                shared_kernel[warp.meta_group_rank()][warp.thread_rank()] = ker_ptr[warp.thread_rank() * ker_sM];
                shared_outg[warp.meta_group_rank()][warp.thread_rank()] = outg_ptr[warp.thread_rank() * outg_sM];
                active_warp.sync();

                for (index_t jj = 0; jj < jj_end; jj++) {
                    scalar_t m2_val = m2_ptr[jj * m2_sM];
                    m1_g_acc += (m1_val - m2_val) * shared_kernel[jj] * shared_outg[jj];
                }
            }
            scalar_t *m1g_ptr = m1_grad.data + i * m1g_sN + k;
            m1g_ptr[0] = m1_g_acc;
        }
    }
}

template <typename scalar_t>
void rbfk_forward_kernel(const at::Tensor &m1,
                         const at::Tensor &m2,
                         const at::Tensor &s,
                         at::Tensor &out)
{
    const auto m1_divs = m1.div(s);
    const auto m2_divs = m2.div(s);
    const auto m1_norm = falkon::ops::square_norm(m1_divs, /*dim=*/1, /*keepdim=*/true);
    const auto m2_norm = falkon::ops::square_norm(m2_divs, /*dim=*/1, /*keepdim=*/true);

    out.copy_(m1_norm.expand_as(out));

    using opmath_t = at::opmath_type<scalar_t>;
    at::addmm_out(out, out, m1_divs, m2_divs.transpose(-2, -1), opmath_t(1), opmath_t(-2));
    out.add_(m2_norm.transpose(-2, -1).expand_as(out));
    out.clamp_min_(opmath_t(1e-20));
    out.mul_(opmath_t(-0.5));
    out.exp_();
}

at::Tensor launch_rbfk_out_forward_kernel(const at::Tensor     & m1,
                                          const at::Tensor     & m2,
                                          const at::Tensor     & s,
                                                at::Tensor     & out)
{
    AT_DISPATCH_FLOATING_TYPES(m1.scalar_type(), "rbf_kernel", [&] {
        rbfk_forward_kernel<scalar_t>(m1, m2, s, out);
    });
    return out;
}
at::Tensor launch_rbfk_forward_kernel(const at::Tensor         & m1,
                                      const at::Tensor         & m2,
                                      const at::Tensor         & s)
{
    at::Tensor out = at::empty({m1.size(0), m2.size(0)}, m1.options());
    AT_DISPATCH_FLOATING_TYPES(m1.scalar_type(), "rbf_kernel", [&] {
        rbfk_forward_kernel<scalar_t>(m1, m2, s, out);
    });
    return out;
}

void launch_rbfk_backward_kernel(const TensorBase &m1,
                                 const TensorBase &m1_grad,
                                 const TensorBase &m2,
                                 const TensorBase &m2_grad,
                                 const TensorBase &s,
                                 const TensorBase &s_grad,
                                 const TensorBase &ker,
                                 const TensorBase &out_grad,
                                 bool m1_requires_grad,
                                 bool m2_requires_grad,
                                 bool s_requires_grad)
{
    auto N = m1.size(0);
    auto D = m1.size(1);
    int64_t count_m1_grad = N * ((D + 31) & -32);  // round up to multiple of 32
    if (count_m1_grad > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(m1.scalar_type(), "rbf_kernel_m1_backward_cuda", [&] {
            if (canUse32BitIndexMath(m1) && canUse32BitIndexMath(m2) && canUse32BitIndexMath(ker)) {
                rbfk_backward_kernel<scalar_t>
                  <<<GET_BLOCKS(count_m1_grad, THREAD_BLOCK_DIM), THREAD_BLOCK_DIM, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<int>(count_m1_grad),
                    getTensorInfo<scalar_t, int>(m1),
                    m1_requires_grad ? getTensorInfo<scalar_t, int>(m1_grad) : TensorInfo<scalar_t, int>(),
                    getTensorInfo<scalar_t, int>(m2),
                    m2_requires_grad ? getTensorInfo<scalar_t, int>(m2_grad) : TensorInfo<scalar_t, int>(),
                    getTensorInfo<scalar_t, int>(s),
                    s_requires_grad ? getTensorInfo<scalar_t, int>(s_grad) : TensorInfo<scalar_t, int>(),
                    getTensorInfo<scalar_t, int>(ker),
                    getTensorInfo<scalar_t, int>(out_grad),
                    m1_requires_grad,
                    m2_requires_grad,
                    s_requires_grad);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                rbfk_backward_kernel<scalar_t>
                  <<<GET_BLOCKS(count_m1_grad, THREAD_BLOCK_DIM), THREAD_BLOCK_DIM, 0, at::cuda::getCurrentCUDAStream()>>>(
                    count_m1_grad,
                    getTensorInfo<scalar_t, int64_t>(m1),
                    m1_requires_grad ? getTensorInfo<scalar_t, int64_t>(m1_grad) : TensorInfo<scalar_t, int64_t>(),
                    getTensorInfo<scalar_t, int64_t>(m2),
                    m2_requires_grad ? getTensorInfo<scalar_t, int64_t>(m2_grad) : TensorInfo<scalar_t, int64_t>(),
                    getTensorInfo<scalar_t, int64_t>(s),
                    s_requires_grad ? getTensorInfo<scalar_t, int64_t>(s_grad) : TensorInfo<scalar_t, int64_t>(),
                    getTensorInfo<scalar_t, int64_t>(ker),
                    getTensorInfo<scalar_t, int64_t>(out_grad),
                    m1_requires_grad,
                    m2_requires_grad,
                    s_requires_grad);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
}
} // end anon namespace

TORCH_LIBRARY_IMPL(falkon, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel_out"),
      TORCH_FN(launch_rbfk_out_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel"),
      TORCH_FN(launch_rbfk_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::rbf_kernel_grad"),
      TORCH_FN(launch_rbfk_backward_kernel));
}

} // end namespace ops
} // end namespace falkon

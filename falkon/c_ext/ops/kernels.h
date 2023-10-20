#pragma once

#include <ATen/ATen.h>

namespace falkon {
namespace ops {

at::Tensor rbf_kernel_out(const at::Tensor         & m1,
                      const at::Tensor         & m2,
                      const at::Tensor         & s,
                            at::Tensor         & out);
at::Tensor rbf_kernel(const at::Tensor         & m1,
                      const at::Tensor         & m2,
                      const at::Tensor         & s);
void rbf_kernel_grad(const at::Tensor         & m1,
                           at::Tensor         & m1_grad,
                       const at::Tensor         & m2,
                             at::Tensor         & m2_grad,
                       const at::Tensor         & s,
                             at::Tensor         & s_grad,
                       const at::Tensor         & ker,
                       const at::Tensor &out_g,
                             bool               m1_requires_grad,
                             bool               m2_requires_grad,
                             bool               s_requires_grad);

} // namespace ops
} // namespace falkon

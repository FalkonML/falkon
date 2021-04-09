#pragma once

#include <torch/extension.h>

/*
 * Multiply a triangular portion of a square tensor by a scalar.
 *
 * Parameters
 * ----------
 * A
 *     The input 2D tensor. `A` must be a square CUDA tensor.
 * upper
 *     Selects the triangular portion of the matrix which will be modified.
 * preserve_diag
 *     Whether the diagonal of the input should be left as is. If `preserve_diag` is false,
 *     the diagonal will be multiplied as well.
 * multiplier
 *     The scalar by which the triangle will be multiplied. Set this to 0 if you
 *     want to clear the triangular portion.
 *
 * Returns
 * -------
 * A
 *     A tensor which shares the same memory as the input.
 */
torch::Tensor mul_triang_cuda(torch::Tensor &A,
                              const bool upper,
                              const bool preserve_diag,
                              const double multiplier);

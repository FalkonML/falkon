#pragma once

#include <torch/extension.h>

/*
 * Copy the triangular part of a 2D tensor, to the opposite triangle. Makes the input tensor symmetric.
 *
 * Parameters
 * ----------
 * A
 *     The input 2D tensor. `A` must be a square CUDA tensor.
 * upper
 *     Keyword used to select the 'origin' triangle for the copy. If `upper` is true,
 *     the upper triangle will be copied to the lower triangle; if `upper is false,
 *     the lower triangle will be copied to the upper triangle.
 *
 * Returns
 * -------
 * A
 *     A tensor which shares the same memory as the input.
 */
torch::Tensor copy_triang_cuda(torch::Tensor &A,
                               const bool upper);

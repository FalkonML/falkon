#pragma once

#include <torch/extension.h>

/*
 * Transpose a 2D matrix, storing the output into a new array.
 *
 * Parameters
 * ----------
 * input
 *     The input 2D matrix.
 * output
 *     The output 2D matrix, which must be of the same size as the input.
 *
 * Returns
 * -------
 * output
 *     A tensor which shares the same memory as `output`, and contains the elements of `input`
 *     transposed.
 */
torch::Tensor copy_transpose_cuda(const torch::Tensor &input,
                                  torch::Tensor &output);

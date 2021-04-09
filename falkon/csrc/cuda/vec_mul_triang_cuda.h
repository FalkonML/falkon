#pragma once

#include <torch/extension.h>

/*
 * Multiply a triangular matrix by a vector (element-wise, using broadcasting)
 *
 * Parameters
 * ----------
 * A
 *     The triangular matrix which should be multiplied.
 * v
 *     A vector with the same dimension as one side of A. Each entry of v will be multiplied by one row, or column
 *     of the triangular matrix A.
 * upper
 *     Whether A is upper triangular.
 * side
 *     Whether to multiply rows or columns of A. If side == 0, each row of A will be multiplied with the same entry
 *     from v. If side == 1, each column will be multiplied with the same number.
 *
 * Returns
 * -------
 * output
 *     A tensor which shares the same memory as the input matrix A.
 */
torch::Tensor vec_mul_triang_cuda(torch::Tensor &A,
                                  torch::Tensor &v,
                                  const bool upper,
                                  const int side);

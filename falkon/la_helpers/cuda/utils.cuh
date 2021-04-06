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
torch::Tensor cuda_copy_triang(torch::Tensor &A,
                               bool upper);


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
torch::Tensor cuda_mul_triang(torch::Tensor &A,
                              bool upper,
                              bool preserve_diag,
                              double multiplier);

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
torch::Tensor cuda_transpose(torch::Tensor &input,
                             torch::Tensor &output);


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
torch::Tensor cuda_vec_mul_triang(torch::Tensor &A,
                                  torch::Tensor &v,
                                  bool upper,
                                  int side);

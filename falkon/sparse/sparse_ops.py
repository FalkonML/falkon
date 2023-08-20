import warnings
from typing import Optional

import torch

from falkon.c_ext import sparse_bdot, sparse_row_norm, sparse_row_norm_sq
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import check_same_dtype
from falkon.utils.tensor_helpers import is_f_contig

__all__ = ("sparse_matmul", "sparse_square_norm", "sparse_norm", "bdot")


def _sparse_matmul_cpu(A: SparseTensor, B: SparseTensor, out: torch.Tensor):
    """
    Inputs:
     - A : N x D, CSR matrix
     - B : D x M, CSC matrix
    """

    if not A.is_csr:
        raise ValueError("A must be CSR matrix")
    if not B.is_csc:
        raise ValueError("B must be CSC matrix")

    try:
        # noinspection PyUnresolvedReferences
        from falkon.mkl_bindings.mkl_bind import mkl_lib

        mkl = mkl_lib()
        try:
            A = A.transpose_csc()  # D * N (csc)
            mkl_sp_1 = mkl.mkl_create_sparse(A)
            mkl_sp_2 = mkl.mkl_create_sparse(B)
            mkl.mkl_spmmd(mkl_sp_1, mkl_sp_2, out, transposeA=True)
        finally:
            try:
                # noinspection PyUnboundLocalVariable
                mkl.mkl_sparse_destroy(mkl_sp_1)
            except:  # noqa: E722
                pass
            try:
                # noinspection PyUnboundLocalVariable
                mkl.mkl_sparse_destroy(mkl_sp_2)
            except:  # noqa: E722
                pass
    except ImportError:
        warnings.warn("Failed to load MKL. Using Scipy sparse matrix multiplication instead.")
        As = A.to_scipy(copy=False)
        Bs = B.to_scipy(copy=False)
        out_sp = As.dot(Bs)
        out_sp.todense(out=out.numpy())
    return out


def _sparse_matmul_cuda(A: SparseTensor, B: SparseTensor, out: torch.Tensor):
    """
    Typically D is very large and since `B` must be in CSR format, memory usage will be quite high.

    Parameters
    ----------
    A : SparseTensor
        N x D :class:`SparseTensor`. Must be in CSR format.
    B : SparseTensor
        D x M :class:`SparseTensor`. Must be in CSR format.
    out : torch.Tensor
        Dense N x M output tensor. Must be F-contiguous (column-contiguous)

    Notes
    ------
    This function runs in two steps:
    sparse*sparse->sparse multiplication and conversion of the output
    sparse matrix to a dense matrix.
    """
    from falkon.c_ext import csr2dense, spspmm

    if not A.is_csr:
        raise ValueError("A must be CSR matrix")
    if not B.is_csr:
        raise ValueError("B must be CSR matrix")
    if not is_f_contig(out, strict=False):
        raise ValueError("out must be F-contiguous")

    # 1. MatMul
    out_indexptr, out_index, out_data = spspmm(A.indexptr, A.index, A.data, B.indexptr, B.index, B.data, B.shape[1])
    # 2. Convert to dense
    out = csr2dense(out_indexptr, out_index, out_data, out)
    return out


def sparse_matmul(A: SparseTensor, B: SparseTensor, out: torch.Tensor) -> torch.Tensor:
    """Sparse*Sparse matrix multiplication. Output will be copied into dense `out` matrix.

    This function can be applied to CPU or CUDA tensors (but all tensors must
    be  on the same device).

    Parameters
    ----------
    A : SparseTensor
        N x D, sparse matrix.
    B : SparseTensor
        D x M, sparse matrix
    out : torch.Tensor
        Dense N x M tensor, it will hold the output of the multiplication.

    Returns
    -------
    out : torch.Tensor
        The same tensor as the input `out` parameter.

    """
    if A.nnz() == 0 or B.nnz() == 0:
        out.fill_(0.0)
        return out

    if A.is_cuda:
        return _sparse_matmul_cuda(A, B, out)
    else:
        return _sparse_matmul_cpu(A, B, out)


def sparse_square_norm(A: SparseTensor, out: torch.Tensor) -> torch.Tensor:
    """Row-wise squared l2 norm of a sparse 2D matrix.

    The operation is equivalent to squaring all elements of the matrix, and summing up the rows.

    Parameters
    ----------
    A : SparseTensor
        The 2D matrix. Since we compute row-wise norms, the matrix must be in CSR format (for
        efficiency).
    out : torch.Tensor
        A dense tensor with the same number of rows as matrix `A`. Will contain the output
        of the squared-norm operation.

    Returns
    -------
    out : torch.Tensor
        The same tensor as the input `out` parameter.

    Notes
    -----
    This function is currently limited to CPU input tensors.
    """
    if out is None:
        out = torch.empty(A.shape[0], 1, dtype=A.dtype, device=A.device)
    if not A.is_csr:
        raise RuntimeError("Sparse squared norm can only be applied on CSR tensors.")
    if not check_same_dtype(A, out):
        raise ValueError("All data-types must match.")
    if A.shape[0] != out.shape[0]:
        raise ValueError("Dimension 0 of A must match the length of tensor 'out'.")

    return sparse_row_norm_sq(A.indexptr, A.data, out=out)


def sparse_norm(A: SparseTensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    """Row-wise l2 norm of a sparse 2D matrix

    Parameters
    ----------
    A : SparseTensor
        The 2D matrix. Since we compute row-wise norms, the matrix must be in CSR format (for
        efficiency).
    out : torch.Tensor
        A dense tensor with the same number of rows as matrix `A`. Will contain the output
        of the norm operation.

    Returns
    -------
    out : torch.Tensor
        The same tensor as the input `out` parameter.

    Notes
    -----
    This function is currently limited to CPU input tensors.
    """
    if out is None:
        out = torch.empty(A.shape[0], 1, dtype=A.dtype, device=A.device)
    if not A.is_csr:
        raise RuntimeError("Sparse norm can only be applied on CSR tensors.")
    if not check_same_dtype(A, out):
        raise ValueError("All data-types must match.")
    if A.shape[0] != out.shape[0]:
        raise ValueError("Dimension 0 of A must match the length of tensor 'out'.")

    return sparse_row_norm(A.indexptr, A.data, out=out)


def bdot(A: SparseTensor, B: SparseTensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    """

    Parameters
    ----------
    A
    B
    out

    Returns
    -------

    """
    if A.shape[0] != B.shape[0]:
        raise RuntimeError("Batch dot can only be applied to matrices with the same number of rows.")
    if out is None:
        out = torch.empty(A.shape[0], 1, dtype=A.dtype, device=A.device)
    if not (A.is_csr and B.is_csr):
        raise RuntimeError("Batch dot can only be applied on CSR tensors.")
    if A.is_cuda or B.is_cuda:
        raise NotImplementedError("Batch dot has not been implemented for sparse CUDA tensors")
    if not check_same_dtype(A, B, out):
        raise ValueError("All data-types must match.")
    if A.shape[0] != out.shape[0]:
        raise ValueError("Output shape must match the number of rows in the input matrices.")

    return sparse_bdot(A.indexptr, A.index, A.data, B.indexptr, B.index, B.data, out=out)

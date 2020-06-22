from typing import Optional

import torch
from falkon.sparse.sparse_helpers import norm_sq, norm_

from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.helpers import check_same_dtype

__all__ = ("sparse_matmul", "sparse_square_norm", "sparse_norm")


def _sparse_matmul_cpu(A, B, out):
    """
    Inputs:
     - A : N x D, CSR matrix
     - B : D x M, CSC matrix
    """
    from falkon.mkl_bindings.mkl_bind import mkl_lib

    if A.nnz() == 0 or B.nnz() == 0:
        return out
    if not A.is_csr:
        raise ValueError("A must be CSR matrix")
    if not B.is_csc:
        raise ValueError("B must be CSC matrix")

    mkl = mkl_lib()
    try:
        # For some reason assigning the 'to_scipy' to their own variables
        # is **absolutely fundamental** for the mkl bindings to work
        A = A.transpose_csc()
        As = A.to_scipy()  # D * N (csc)
        Bs = B.to_scipy()

        mkl_sp_1 = mkl.mkl_create_sparse_from_scipy(As)
        mkl_sp_2 = mkl.mkl_create_sparse_from_scipy(Bs)
        mkl.mkl_spmmd(mkl_sp_1, mkl_sp_2, out, transposeA=True)
        return out
    finally:
        try:
            # noinspection PyUnboundLocalVariable
            mkl.mkl_sparse_destroy(mkl_sp_1)
        except:
            pass
        try:
            # noinspection PyUnboundLocalVariable
            mkl.mkl_sparse_destroy(mkl_sp_2)
        except:
            pass


def _sparse_matmul_cuda(A: SparseTensor, B: SparseTensor, out: torch.Tensor):
    """
    Typically D is very large, and we will need to convert B to CSR format
    so memory usage will be high.

    Parameters
    ----------
    A : N x D, CSR matrix
    B : D x M, CSR matrix

    Notes
    ------
    This function runs in two steps:
    sparse*sparse->sparse multiplication and conversion of the output
    sparse matrix to a dense matrix.
    """
    from falkon.sparse.sparse_helpers import spspmm, csr2dense

    if not A.is_csr:
        raise ValueError("A must be CSR matrix")
    if not B.is_csr:
        raise ValueError("B must be CSR matrix")

    # 2. MatMul
    out_indexptr, out_index, out_data = spspmm(
        A.indexptr, A.index, A.data, B.indexptr, B.index, B.data, A.shape[1])
    # 3. Convert to dense
    out = csr2dense(out_indexptr, out_index, out_data, out)
    return out


def sparse_matmul(A: SparseTensor, B: SparseTensor, out: torch.Tensor) -> torch.Tensor:
    """Sparse*Sparse matrix multiplication. Output will be copied into dense `out` matrix.

    This function can be applied to CPU or CUDA tensors (but all tensors must
    be consistently on the same device). Note that the CUDA matrix multiplication
    is

    Parameters
    ----------
    A : SparseTensor
        N x D, sparse matrix.
    B : SparseTensor
        D x M, sparse matrix

    """
    if A.nnz() == 0 or B.nnz() == 0:
        return out

    if A.is_cuda:
        return _sparse_matmul_cuda(A, B, out)
    else:
        return _sparse_matmul_cpu(A, B, out)


def sparse_square_norm(A: SparseTensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if not A.is_csr:
        raise RuntimeError("Squared norm can only be applied on CSR tensors")
    if not check_same_dtype(A, out):
        raise ValueError("All data-types must match")
    if A.shape[0] != out.shape[0]:
        raise ValueError("Dimension 0 of A must match the length of tensor 'out'")

    return norm_sq(A.indexptr, A.data, out)


def sparse_norm(A: SparseTensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if not A.is_csr:
        raise RuntimeError("Norm can only be applied on CSR tensors")
    if not check_same_dtype(A, out):
        raise ValueError("All data-types must match")
    if A.shape[0] != out.shape[0]:
        raise ValueError("Dimension 0 of A must match the length of tensor 'out'")

    return norm_(A.indexptr, A.data, out)
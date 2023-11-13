"""Wrap the the various linear-algebra helpers which use c extension
"""
from typing import Optional

import torch

from falkon import c_ext
from falkon.la_helpers.cpu_trsm import cpu_trsm
from falkon.utils.helpers import check_same_device

__all__ = (
    "zero_triang",
    "mul_triang",
    "copy_triang",
    "vec_mul_triang",
    "potrf",
    "trsm",
    "square_norm",
)


def zero_triang(mat: torch.Tensor, upper: bool) -> torch.Tensor:
    """Set the upper/lower triangle of a square matrix to 0.

    Note that the diagonal will be preserved, and that this function
    operates in-place.

    Parameters
    ----------
    mat
        The input 2D tensor. This can also be a CUDA tensor.
    upper
        Whether to zero-out the upper, or the lower triangular part of `mat`.

    Returns
    -------
    mat
        The same tensor as was passed as a input with the upper or lower triangle zeroed-out.
    """
    return c_ext.mul_triang(mat, upper=upper, preserve_diag=True, multiplier=0.0)


def mul_triang(mat: torch.Tensor, upper: bool, preserve_diag: bool, multiplier: float) -> torch.Tensor:
    """Multiply a triangular matrix by a scalar.

    The input is a square matrix, and parameters determine what exactly is the triangular
    part which will be multiplied. CUDA and CPU tensors as well as numpy arrays
    are supported.
    This operation runs in-place.

    Parameters
    ----------
    mat
        The input square tensor. This can also be a CUDA tensor.
    upper
        Whether to consider the upper, or the lower triangular part of `mat`.
    preserve_diag
        Whether the diagonal of `mat` will be multiplied. If `preserve_diag=True`, then the
        diagonal will not be multiplied.
    multiplier
        The scalar by which the triangular input matrix will be multiplied.

    Returns
    -------
    mat
        The same tensor as was passed as input, with the desired operation performed on it.
    """
    return c_ext.mul_triang(mat, upper=upper, preserve_diag=preserve_diag, multiplier=multiplier)


def copy_triang(mat: torch.Tensor, upper: bool) -> torch.Tensor:
    """Copy one triangle of `mat` to the other, making it symmetric.

    The input is a square matrix: CUDA and CPU tensors as well as numpy arrays are supported.
    This operation runs in-place.

    Parameters
    ----------
    mat
        The input square tensor, or numpy array. This can also be a CUDA tensor.
    upper
        If `upper=True` the upper triangle will be copied into the lower triangle of `mat`,
        otherwise the lower triangle of `mat` will be copied into its upper triangle.

    Returns
    -------
    mat
        The same tensor, with the desired operation performed on it.
        The output matrix will be symmetric.
    """
    return c_ext.copy_triang(mat, upper=upper)


def vec_mul_triang(mat: torch.Tensor, multipliers: torch.Tensor, upper: bool, side: int) -> torch.Tensor:
    multipliers = multipliers.reshape(-1)
    return c_ext.vec_mul_triang(mat, multipliers, upper=upper, side=side == 1)


def potrf(mat: torch.Tensor, upper: bool, clean: bool, overwrite: bool, cuda: bool) -> torch.Tensor:
    if mat.is_cuda or cuda:
        raise NotImplementedError(
            "'potrf' is only implemented for CPU tensors. See the ooc_ops module for CUDA implementations."
        )
    return c_ext.potrf(mat, upper=upper, clean=clean, overwrite=overwrite)


def trsm(v: torch.Tensor, A: torch.Tensor, alpha: float, lower: int = 0, transpose: int = 0) -> torch.Tensor:
    if isinstance(A, torch.Tensor):
        if isinstance(v, torch.Tensor):
            if not check_same_device(A, v):
                raise ValueError("A and v must be on the same device.")
            if A.is_cuda and v.is_cuda:
                from falkon.la_helpers.cuda_trsm import cuda_trsm

                return cuda_trsm(A, v, alpha, bool(lower), bool(transpose))
            else:
                A = A.numpy()
                v = v.numpy()
        else:  # v is numpy array (thus CPU)
            if A.is_cuda:
                raise ValueError("A and v must be on the same device.")
            A = A.numpy()

    vout = cpu_trsm(A, v, alpha, lower, transpose)
    return torch.from_numpy(vout)

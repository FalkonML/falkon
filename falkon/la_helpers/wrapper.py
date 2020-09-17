"""Wrap the the various linear-algebra helpers which use
1. cython (for numpy arrays)
2. torch-extension (for cuda tensors)
In order to provide a unified interface
"""
from typing import Union

import numpy as np
import torch

from falkon.la_helpers.cyblas import (
    vec_mul_triang as c_vec_mul_triang,
    mul_triang as c_mul_triang,
    copy_triang as c_copy_triang,
    potrf as c_potrf
)
from falkon.utils import decide_cuda
from falkon.utils.helpers import check_same_device
from .cpu_trsm import cpu_trsm

if decide_cuda():
    # noinspection PyUnresolvedReferences
    from falkon.la_helpers.cuda_la_helpers import (
        cuda_copy_triang,
        cuda_mul_triang
    )
    from .cuda_trsm import cuda_trsm

arr_type = Union[torch.Tensor, np.ndarray]
__all__ = ("zero_triang", "mul_triang", "copy_triang", "vec_mul_triang", "potrf", "trsm")


def zero_triang(mat: arr_type, upper: bool) -> arr_type:
    """Set the upper/lower triangle of a square matrix to 0.

    Note that the diagonal will be preserved, and that this function
    operates in-place.

    Parameters
    ----------
    mat
        The input 2D tensor, or numpy array. This can also be a CUDA tensor.
    upper
        Whether to zero-out the upper, or the lower triangular part of `mat`.

    Returns
    -------
    mat
        The same tensor, or numpy array as was passed as a parameter,
        with the upper or lower triangle zeroed-out.
    """
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            return cuda_mul_triang(mat, upper=upper, preserve_diag=True, multiplier=0.0)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_mul_triang(mat, upper=upper, preserve_diag=1, multiplier=0.0)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def mul_triang(mat: arr_type, upper: bool, preserve_diag: bool, multiplier: float) -> arr_type:
    """Multiply a triangular matrix by a scalar.

    The input is a square matrix, and parameters determine what exactly is the triangular
    part which will be multiplied. CUDA and CPU tensors as well as numpy arrays
    are supported.
    This operation runs in-place.

    Parameters
    ----------
    mat
        The input square tensor, or numpy array. This can also be a CUDA tensor.
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
        The same tensor, or numpy array as was passed as a parameter, with the desired
        operation performed on it.
    """
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            s1 = torch.cuda.Stream(device=mat.device)
            try:
                with torch.cuda.stream(s1):
                    return cuda_mul_triang(mat, upper=upper, preserve_diag=preserve_diag, multiplier=multiplier)
            finally:
                s1.synchronize()
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_mul_triang(mat, upper=upper, preserve_diag=int(preserve_diag), multiplier=multiplier)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def copy_triang(mat: arr_type, upper: bool) -> arr_type:
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
        The same tensor, or numpy array as was passed as a parameter, with the desired
        operation performed on it. The output matrix will be symmetric.
    """
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            s1 = torch.cuda.Stream(device=mat.device)
            try:
                with torch.cuda.stream(s1):
                    return cuda_copy_triang(mat, upper=upper)
            finally:
                s1.synchronize()
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_copy_triang(mat, upper=upper)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def vec_mul_triang(mat: arr_type, multipliers: arr_type, upper: bool, side: int) -> arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            raise NotImplementedError("'vec_mul_triang' is only implemented for CPU tensors")
        else:
            out_torch_convert = True
            mat = mat.numpy()
    if isinstance(multipliers, torch.Tensor):
        multipliers = multipliers.numpy()
    out = c_vec_mul_triang(mat, multiplier=multipliers, upper=upper, side=side)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def potrf(mat: arr_type, upper: bool, clean: bool, overwrite: bool, cuda: bool) -> arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda or cuda:
            raise NotImplementedError("'potrf' is only implemented for CPU tensors. "
                                      "See the ooc_ops module for CUDA implementations.")
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_potrf(mat, upper=upper, clean=clean, overwrite=overwrite)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def trsm(v: arr_type, A: arr_type, alpha: float, lower: int = 0, transpose: int = 0) -> arr_type:
    out_torch_convert = False
    if isinstance(A, torch.Tensor):
        if isinstance(v, torch.Tensor):
            if not check_same_device(A, v):
                raise ValueError("A and v must be on the same device.")
            if A.is_cuda and v.is_cuda:
                return cuda_trsm(A, v, alpha, lower, transpose)
            else:
                out_torch_convert = True
                A = A.numpy()
                v = v.numpy()
        else:  # v is numpy array (thus CPU)
            if A.is_cuda:
                raise ValueError("A and v must be on the same device.")
            else:
                out_torch_convert = True
                A = A.numpy()

    vout = cpu_trsm(A, v, alpha, lower, transpose)
    if out_torch_convert:
        return torch.from_numpy(vout)
    return vout

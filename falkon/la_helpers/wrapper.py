"""Wrap the the various linear-algebra helpers which use
1. cython (for numpy arrays)
2. torch-extension (for cuda tensors)
In order to provide a unified interface
"""
from typing import Union

import numpy as np
import torch
from scipy.linalg import blas as sclb

from falkon.la_helpers.cyblas import (
    vec_mul_triang as c_vec_mul_triang,
    mul_triang as c_mul_triang,
    copy_triang as c_copy_triang,
    potrf as c_potrf
)
from falkon.utils.helpers import check_same_device, choose_fn
from falkon.utils import decide_cuda
if decide_cuda():
    from falkon.la_helpers.cuda_la_helpers import (
        cuda_copy_triang,
        cuda_mul_triang
    )

arr_type = Union[torch.Tensor, np.ndarray]
__all__ = ("zero_triang", "mul_triang", "copy_triang", "vec_mul_triang", "potrf", "trsm")


def zero_triang(mat: arr_type, upper: bool) -> arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            print("cuda_mul_triang(", mat.size(), "upper", upper)
            return cuda_mul_triang(mat, upper=upper, preserve_diag=True, multiplier=0.0)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_mul_triang(mat, upper=upper, preserve_diag=1, multiplier=0.0)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def mul_triang(mat: arr_type, upper: bool, preserve_diag: bool, multiplier: float) -> arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            return cuda_mul_triang(mat, upper=upper, preserve_diag=preserve_diag, multiplier=multiplier)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_mul_triang(mat, upper=upper, preserve_diag=int(preserve_diag), multiplier=multiplier)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def copy_triang(mat: arr_type, upper: bool) -> arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            return cuda_copy_triang(mat, upper=upper)
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
            if A.is_cuda and v.is_cuda:
                raise NotImplementedError("'trsm' is only implemented for CPU tensors.")
            elif not check_same_device(A, v):
                raise ValueError("A and v must be on the same device.")
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

    # Run the CPU version of TRSM. Now everything is numpy.
    trsm_fn = choose_fn(A.dtype, sclb.dtrsm, sclb.strsm, "TRSM")
    vF = np.copy(v, order='F')
    trsm_fn(alpha, A, vF,  side=0, lower=lower, trans_a=transpose, overwrite_b=1)
    if not v.flags.f_contiguous:
        vF = np.copy(vF, order='C')

    if out_torch_convert:
        return torch.from_numpy(vF)
    return vF

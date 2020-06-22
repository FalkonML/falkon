import functools

import numpy as np
import torch
from scipy.linalg import blas as sclb, lapack as scll

from falkon.options import FalkonOptions
from falkon.utils.cyblas import potrf
from falkon.utils.helpers import choose_fn

__all__ = ("check_init", "trsm", "inplace_set_diag", "inplace_add_diag", "lauum_wrapper",
           "potrf_wrapper",)


def check_init(*none_check):
    def _checker(fun):
        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            is_init = True
            for el in none_check:
                if getattr(self, el, None) is None:
                    is_init = False
                    break
            if not is_init:
                raise RuntimeError(
                    "FALKON preconditioner is not initialized. Please run "
                    "`init` before any other method on the "
                    "preconditioner.")
            return fun(self, *args, **kwargs)
        return wrapper
    return _checker


def trsm(v, A, alpha, lower=0, transpose=0):
    """Solve triangular system Ax = v
    """
    trsm_fn = choose_fn(A.dtype, sclb.dtrsm, sclb.strsm, "TRSM")
    vF = np.copy(v.numpy(), order='F')
    trsm_fn(alpha, A, vF,
            side=0, lower=lower, trans_a=transpose, overwrite_b=1)
    if not v.numpy().flags.f_contiguous:
        vF = np.copy(vF, order='C')
    return torch.from_numpy(vF)


def inplace_set_diag(A, k):
    # Assumes M is square (or wide also works).
    # Look at np.fill_diagonal
    step = A.shape[1] + 1
    A.flat[::step] = k
    return A


def inplace_add_diag(A, k):
    # Assumes M is square (or wide also works).
    # Look at np.fill_diagonal
    step = A.shape[1] + 1
    A.flat[::step] += k
    return A


def lauum_wrapper(A: np.ndarray, upper: bool, use_cuda: bool, opt: FalkonOptions) -> np.ndarray:
    if use_cuda:
        from falkon.ooc_ops.ooc_lauum import gpu_lauum
        return gpu_lauum(A, upper=upper, write_opposite=True, overwrite=True, opt=opt)
    else:
        lauum = choose_fn(A.dtype, scll.dlauum, scll.slauum, "LAUUM")
        sol, info = lauum(A, lower=int(not upper), overwrite_c=1)
        if info != 0:
            raise RuntimeError(f"Lapack LAUUM failed with error code {info}.")
        return sol


def potrf_wrapper(A: np.ndarray, clean: bool, upper: bool, use_cuda: bool, opt: FalkonOptions) -> np.ndarray:
    if use_cuda:
        from falkon.ooc_ops.ooc_potrf import gpu_cholesky
        return gpu_cholesky(torch.from_numpy(A), upper=upper, clean=clean, overwrite=True, opt=opt).numpy()
    else:
        return potrf(A, upper=upper, clean=clean, overwrite=True)

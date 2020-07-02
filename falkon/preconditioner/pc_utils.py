import functools

import numpy as np
import torch
from scipy.linalg import blas as sclb, lapack as scll

from falkon.options import FalkonOptions
from falkon.utils.cyblas import potrf
from falkon.utils.helpers import choose_fn

__all__ = ("check_init", "trsm", "inplace_set_diag", "inplace_add_diag", "inplace_add_diag_th", 
           "lauum_wrapper", "potrf_wrapper",)


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


def inplace_add_diag_th(A: torch.Tensor, k) -> torch.Tensor:
    # Assumes M is square (or wide also works).
    # Need to use .diagonal() as .diag() makes a copy
    d = A.diagonal()
    d += k
    return A


def lauum_wrapper(A: torch.Tensor, upper: bool, use_cuda: bool, opt: FalkonOptions) -> torch.Tensor:
    if use_cuda:
        from falkon.ooc_ops.ooc_lauum import gpu_lauum
        return gpu_lauum(A, upper=upper, write_opposite=True, overwrite=True, opt=opt)
    else:
        Anp = A.numpy()
        lauum = choose_fn(Anp.dtype, scll.dlauum, scll.slauum, "LAUUM")
        sol, info = lauum(Anp, lower=int(not upper), overwrite_c=1)
        if info != 0:
            raise RuntimeError(f"Lapack LAUUM failed with error code {info}.")
        return torch.from_numpy(sol)


def potrf_wrapper(A: torch.Tensor, clean: bool, upper: bool, use_cuda: bool, opt: FalkonOptions) -> torch.Tensor:
    if use_cuda:
        from falkon.ooc_ops.ooc_potrf import gpu_cholesky
        return gpu_cholesky(A, upper=upper, clean=clean, overwrite=True, opt=opt)
    else:
        return torch.from_numpy(potrf(A.numpy(), upper=upper, clean=clean, overwrite=True))

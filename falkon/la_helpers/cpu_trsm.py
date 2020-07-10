import numpy as np
from scipy.linalg import blas as sclb

from falkon.utils.helpers import choose_fn


def cpu_trsm(A: np.ndarray, v: np.ndarray, alpha: float, lower: int, transpose: int) -> np.ndarray:
    # Run the CPU version of TRSM. Now everything is numpy.
    trsm_fn = choose_fn(A.dtype, sclb.dtrsm, sclb.strsm, "TRSM")
    vF = np.copy(v, order='F')
    trsm_fn(alpha, A, vF,  side=0, lower=lower, trans_a=transpose, overwrite_b=1)
    if not v.flags.f_contiguous:
        vF = np.copy(vF, order='C')
    return vF

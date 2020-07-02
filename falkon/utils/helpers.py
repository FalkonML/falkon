import math
from typing import Optional, Union, List, Type

import numpy as np
import torch
import torch.multiprocessing

from falkon.sparse.sparse_tensor import SparseTensor


__all__ = ("check_sparse", "select_dim_fMM", "check_same_device",
           "select_dim_over_d", "select_dim_over_m", "calc_gpu_block_sizes", "choose_fn", "sizeof_dtype", "check_same_dtype",
           )


def check_sparse(*args: Union[torch.Tensor, SparseTensor]) -> List[bool]:
    out = []
    for t in args:
        out.append(isinstance(t, SparseTensor))
    return out


def select_dim_fMM(tot, maxN, maxD, maxM):
    """Calculate the maximum block size given a maximum amount of memory.

    Parameters
    -----------
    tot : float
        The maximal amount of memory that can be used.
        This constrains the solution since $n*d + M*d + n*M <= tot$
    maxN : int
        The maximal value of the `N` dimension.
    maxD : int
        The maximal value of the `D` dimension.
    maxM : int
        The maximal value of the `M` dimension.

    Returns
    --------
    blockN, blockM, blockD : int, int, int
        The maximum block dimensions which satisfy the provided
        memory constraints.

    Notes
    ------
    solves the problem, max ndM such that n <= maxN, d <= maxD, M <= maxM
    nd + Md + nM <= tot
    """
    order, ind = torch.tensor([maxN, maxM, maxD], dtype=torch.float64).sort()
    vlx = torch.ones(3, dtype=torch.float64) * math.sqrt(tot / 3)

    vlx = order.min(vlx)
    # noinspection PyTypeChecker
    vlx[1] = min(order[1], math.sqrt(tot + vlx[0]**2) - vlx[0])
    vlx[2] = min(order[2], (tot - vlx[0] * vlx[1]) / (vlx[0] + vlx[1]))

    vlx = vlx[ind]

    bN, bM, bD = int(vlx[0]), int(vlx[1]), int(vlx[2])
    if bN <= 0 or bM <= 0 or bD <= 0:
        raise RuntimeError("Available memory %.2fMB is not enough." % (tot / 2**20))

    return bN, bM, bD


def select_dim_over_d(maxD, maxN, coef_nd, coef_n, coef_d, rest, tot):
    """
    solves the problem, max n*d such that n <= maxN, d <= maxD and
    coef_nd*nd + coef_n*n + coef_d*d + rest <= tot
    """
    tot = tot - rest

    if coef_nd == 0:
        # We have a linear problem: coef_n*n + coef_d*d <= tot
        # for now we just solve this in a bad way TODO
        coef_nd = 1e-10

    b = coef_n + coef_d
    x = (-b + math.sqrt(b**2 +4*coef_nd*tot)) / (2*coef_nd)
    d = math.floor(min(maxD, x))
    n = math.floor(min(maxN, x))

    if d == maxD and n < maxN:
        n = (tot - coef_d*d) / (coef_nd*d + coef_n)
        n = min(maxN, n)
    elif d < maxD and n == maxN:
        d = (tot - coef_n*n) / (coef_nd*n + coef_d)
        d = min(maxD, d)

    n, d = int(n), int(d)
    if n <= 0 or d <= 0:
        raise RuntimeError("Available memory %.2fMB is not enough." % ((tot + rest) / 2**20))
    return n, d


def select_dim_over_m(maxM, maxN, coef_nm, coef_n, coef_m, tot, rest=0):
    """
    solves the problem, max n*m such that n <= maxN, m <= maxM and
    coef_nm*nm + coef_n*n + coef_m*m <= tot
    """
    tot = tot - rest
    # We consider x = m = n and solve the quadratic equation
    b = coef_n + coef_m
    x = (-b + math.sqrt(b**2 + 4*coef_nm*tot)) / (2 * coef_nm)
    m = math.floor(min(maxM, x))
    n = math.floor(min(maxN, x))

    # If one of the two n, m was capped at it's limit we want to
    # recalculate the value of the other variable by solving the
    # corresponding linear equation.
    if m == maxM and n < maxN:
        n = (tot - coef_m * m) / (coef_nm * m + coef_n)
        n = min(maxN, n)
    if n == maxN and m < maxM:
        m = (tot - coef_n * n) / (coef_nm * n + coef_m)
        m = min(maxM, m)

    n, m = int(n), int(m)
    if n <= 0 or m <= 0:
        raise RuntimeError("Available memory %.2fMB is not enough." % (tot / 2**20))
    return n, m


def calc_gpu_block_sizes(device_info, tot_size):
    gpu_speed = np.array([g.speed for g in device_info])
    speed_frac = np.array(gpu_speed) / np.sum(gpu_speed)

    block_sizes = np.cumsum(np.concatenate(([0], speed_frac))) * tot_size
    block_sizes[0] = 0
    block_sizes[-1] = tot_size

    return np.floor(block_sizes).astype(np.int64).tolist()


def choose_fn(dtype, f64_fn, f32_fn, fn_name):
    # Necessary to check torch early because comparing
    # torch.dtype == numpy.dtype results in a type-error.
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            return f64_fn
        if dtype == torch.float32:
            return f32_fn
    if dtype == np.float64:
        return f64_fn
    if dtype == np.float32:
        return f32_fn

    raise TypeError("No %s function exists for data type %s." % (fn_name, dtype))


def sizeof_dtype(dtype: Union[torch.dtype, np.dtype, Type]) -> int:
    # Necessary to check torch early because comparing
    # torch.dtype == numpy.dtype results in a type-error.
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            return 8
        if dtype == torch.float32:
            return 4
    if dtype == np.float64:
        return 8
    if dtype == np.float32:
        return 4

    raise TypeError("Dtype %s not valid" % (dtype))


def check_same_dtype(*args: Optional[Union[torch.Tensor, SparseTensor]]) -> bool:
    dt = None
    all_equal = True

    for a in args:
        if a is None:
            continue
        if dt is None:
            dt = a.dtype
        else:
            all_equal &= a.dtype == dt
    return all_equal


def check_same_device(*args: Union[None, torch.Tensor, SparseTensor]) -> bool:
    dev = None
    for t in args:
        if t is None:
            continue
        t_dev = t.device
        if dev is None:
            dev = t_dev
        elif t_dev != dev:
            return False
    return True

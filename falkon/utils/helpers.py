import math
from typing import List, Optional, Type, Union

import numpy as np
import torch
import torch.multiprocessing

from falkon.sparse.sparse_tensor import SparseTensor

__all__ = (
    "select_dim_over_n",
    "select_dim_over_bnm",
    "select_dim_over_nm",
    "select_dim_over_nd",
    "select_dim_over_nm_v2",
    "calc_gpu_block_sizes",
    "choose_fn",
    "sizeof_dtype",
    "check_sparse",
    "check_same_dtype",
    "check_same_device",
)


def solve_quad(a, b, c):
    if a == 0:
        return float("inf")
    return (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)


def solve_lin(b, c):
    return -c / b


def select_dim_over_bnm(max_b, max_n, max_m, d, coef_bnd, coef_bmd, coef_bnm, coef_bn, coef_bm, rest, max_mem):
    def calc_used_mem(b_, n_, m_):
        return (
            coef_bnd * b_ * n_ * d
            + coef_bmd * b_ * m_ * d
            + coef_bnm * b_ * n_ * m_
            + coef_bn * b_ * n_
            + coef_bm * b_ * m_
            + rest
        )

    if calc_used_mem(1, max_n, max_m) < max_mem:
        # find min-b which fits
        n, m = max_n, max_m
        b = solve_lin(
            b=(coef_bnd * n * d + coef_bmd * m * d + coef_bnm * n * m + coef_bn * n + coef_bm * m), c=rest - max_mem
        )
    elif calc_used_mem(1, 1, max_m) < max_mem:
        # find min-n which fits
        b, m = 1, max_m
        n = solve_lin(
            b=coef_bnd * b * d + coef_bnm * b * m + coef_bn * b,
            c=rest + coef_bmd * b * m * d + coef_bm * b * m - max_mem,
        )
    else:
        # find min-m which fits
        b, n = 1, 1
        m = solve_lin(
            b=coef_bmd * b * d + coef_bnm * b * n + coef_bm * b, c=rest + coef_bnd * b * n * d + coef_bn * b * n - max_m
        )

    out_b = int(min(b, max_b))
    out_n = int(min(n, max_n))
    out_m = int(min(m, max_m))
    if out_b <= 0 or out_n <= 0 or out_m <= 0:
        raise MemoryError("Available memory %.2fMB is not enough." % (max_mem / 2**20))
    return out_b, out_n, out_m


def select_dim_over_n(max_n, m, d, coef_nm, coef_nd, coef_md, coef_n, coef_m, coef_d, rest, max_mem):
    """
    n * (m * coef_nm + d * coef_nd + coef_n) + rest <= max_mem
    """
    n_coef = m * coef_nm + d * coef_nd + coef_n
    rest_mem = rest + coef_md * m * d + coef_m * m + coef_d * d
    v_n = (max_mem - rest_mem) / n_coef

    out_n = int(min(v_n, max_n))
    if out_n <= 0:
        raise MemoryError("Available memory %.2fMB is not enough." % (max_mem / 2**20))
    return out_n


def select_dim_over_nm(max_n, max_m, d, coef_nd, coef_md, coef_nm, coef_n, coef_m, rest, max_mem):
    """Finds the optimal values for `n` and `m` to fit in available memory.

    This function should be called for problems where the GPU needs to hold
    two blocks of data (one of size m, one of size n) and one kernel block
    (of size n x m).

    Parameters
    -----------
    max_n : int
        The maximum value for n (the first dimension of the problem)
    max_m : int
        The maximum value for m (the second dimension of the problem)
    d : int
        The dimensionality of the data
    coef_nd : float
        How many n*d blocks need to be held in memory
    coef_md : float
        How many m*d blocks need to be held in memory
    coef_nm : float
        How many m*n blocks need to be held in memory
    coef_n : float
        How many n-dimensional vectors need to be held in memory
    coef_m : float
        How many m-dimensional vectors need to be held in memory
    rest : float
        additional bytes to be kept in memory
    max_mem : float
        The amount of available memory in bytes. This is the main problem constraint

    Returns
    -------
    out_n : int
        The dimension n to use in order to fit in available memory
    out_m : int
        The dimension m to use in order to fit in available memory

    Notes
    ------
    The equation gives a hyperbola. We intersect the hyperbola
    with a line from the origin, with the slope given by the ratio
    of max_m and max_n. We then solve a quadratic equation to find
    the intersection point.
    """
    fac = max_m / max_n

    if coef_nm == 0 and (coef_nd == 0 and coef_md == 0 and coef_n == 0 and coef_m == 0):
        v_n = max_n
    elif coef_nm == 0:
        v_n = solve_lin(b=d * (coef_nd + fac * coef_md) + coef_n + coef_m * fac, c=rest - max_mem)
    else:
        v_n = solve_quad(a=fac * coef_nm, b=d * (fac * coef_md + coef_nd) + fac * coef_m + coef_n, c=rest - max_mem)
    v_m = fac * v_n

    out_n = int(min(v_n, max_n))
    out_m = int(min(v_m, max_m))
    if out_n <= 0 or out_m <= 0:
        raise MemoryError("Available memory %.2fMB is not enough." % (max_mem / 2**20))
    return out_n, out_m


def select_dim_over_nd(max_n, max_d, coef_nd, coef_n, coef_d, rest, max_mem):
    """
    solves the problem, max n*d such that n <= maxN, d <= maxD and
    coef_nd*nd + coef_n*n + coef_d*d + rest <= tot
    """
    if coef_nd == 0 and (coef_n == 0 or coef_d == 0):  # One or 0 variables interesting
        if coef_d == coef_n:
            n, d = max_n, max_d
        elif coef_n == 0:
            n = max_n
            d = (max_mem - rest) / coef_d
        else:  # coef_d == 0
            n = (max_mem - rest) / coef_n
            d = max_d
    else:  # Both variables are used. We solve assuming n == d
        if coef_nd == 0:
            x = solve_lin(b=coef_n + coef_d, c=rest - max_mem)
        else:
            try:
                x = solve_quad(a=coef_nd, b=coef_n + coef_d, c=rest - max_mem)
            except ValueError:  # Does not intersect x-axis.
                x = -1
        n = math.floor(min(max_n, x))
        d = math.floor(min(max_d, x))
        # If one of n, d reaches the max, try use up the remaining memory on the other one.
        if d == max_d and n < max_n:
            # Assume d fixed at maxD, and derive for the best value of n
            n = (max_mem - rest - coef_d * d) / (coef_nd * d + coef_n)
        elif d < max_d and n == max_n:
            # Assume n fixed at maxN, and derive for the best value of d
            d = (max_mem - rest - coef_n * n) / (coef_nd * n + coef_d)

    n = int(min(max_n, n))
    d = int(min(max_d, d))
    if n <= 0 or d <= 0:
        raise MemoryError("Available memory %.2fMB is not enough." % (max_mem / 2**20))
    return n, d


def select_dim_over_nm_v2(max_n, max_m, coef_nm, coef_n, coef_m, rest, max_mem):
    """
    solves the problem, max n*m such that n <= maxN, m <= maxM and
    coef_nm*nm + coef_n*n + coef_m*m <= tot
    """
    return select_dim_over_nd(
        max_n=max_n, max_d=max_m, coef_nd=coef_nm, coef_n=coef_n, coef_d=coef_m, rest=rest, max_mem=max_mem
    )


def calc_gpu_block_sizes(device_info, tot_size):
    gpu_speed = np.asarray([g.speed for g in device_info])
    speed_frac = np.asarray(gpu_speed) / np.sum(gpu_speed)

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

    raise TypeError(f"No {fn_name} function exists for data type {dtype}.")


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


def check_sparse(*args: Union[torch.Tensor, SparseTensor]) -> List[bool]:
    return [isinstance(t, SparseTensor) for t in args]


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

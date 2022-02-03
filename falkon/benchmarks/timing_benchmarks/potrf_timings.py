import functools

import time
import json
from typing import List, Dict, Any

import torch
import numpy as np
from scipy.linalg.lapack import spotrf, dpotrf

import falkon
from falkon.ooc_ops.ooc_potrf import gpu_cholesky
from falkon.la_helpers.cyblas import add_symmetrize
from falkon.utils import devices


def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return out.T
    return out


def gen_random_pd(t, dtype, F=False, seed=0):
    A = gen_random(t, t, dtype, F, seed)
    add_symmetrize(A)
    A.flat[::t + 1] += t
    return A


def run_potrf_exp(exp_name, fn, exp_sizes, num_reps, is_torch, dtype):
    timings = []
    for num_pts in exp_sizes:
        A = gen_random_pd(num_pts, dtype, F=True, seed=192)

        rep_times = []
        for j in range(num_reps):
            if is_torch:
                Ac = torch.from_numpy(A.copy(order='F'))
            else:
                Ac = A.copy(order='F')
            t_s = time.time()
            fn(Ac)
            t_e = time.time()
            rep_times.append(t_e - t_s)
            print("Exp %s - N %d - Rep %d - %.2fs" % (exp_name, num_pts, j, rep_times[-1]),
                  flush=True)
            del Ac
            if is_torch:
                torch.cuda.empty_cache()
        timings.append(min(rep_times))
    return timings


if __name__ == "__main__":
    init_opt = falkon.FalkonOptions()
    torch.cuda.init()
    gpu_info = [v for k, v in devices.get_device_info(init_opt).items() if k >= 0]
    num_gpu = len(gpu_info)

    defaultN32 = [10_000, 20_000, 30_000, 40_000, 50_000, 65_000, 80_000, 100_000, 120_000, 140_000]
    defaultN64 = [10_000, 20_000, 30_000, 40_000, 50_000, 65_000, 80_000]
    falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2, compute_arch_speed=False)

    experiments: List[Dict[str, Any]] = [
        {
            'exp_name': 'Parallel 32',
            'exp_sizes': defaultN32,
            'dtype': np.float32,
            'num_reps': 3,
            'is_torch': True,
            'fn': functools.partial(
                gpu_cholesky, upper=False, clean=False, overwrite=True,
                opt=falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2)),
        },
        {
            'exp_name': 'Parallel 64',
            'exp_sizes': defaultN64,
            'dtype': np.float64,
            'num_reps': 3,
            'is_torch': True,
            'fn': functools.partial(
                gpu_cholesky, upper=False, clean=False, overwrite=True,
                opt=falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2,
                                         compute_arch_speed=False)),
        },
        {
            'exp_name': 'CPU 32',
            'exp_sizes': defaultN32,
            'dtype': np.float32,
            'num_reps': 3,
            'is_torch': False,
            'fn': functools.partial(spotrf, lower=True, clean=False, overwrite_a=True),
        },
        {
            'exp_name': 'CPU 64',
            'exp_sizes': defaultN64,
            'dtype': np.float64,
            'num_reps': 2,
            'is_torch': False,
            'fn': functools.partial(dpotrf, lower=True, clean=False, overwrite_a=True),
        },
    ]
    for exp in experiments:
        exp_times = run_potrf_exp(**exp)
        exp['timings'] = exp_times
    with open("logs/potrf_timings_%dGPU.json" % (num_gpu), "w") as fh:
        json.dump(experiments, fh)

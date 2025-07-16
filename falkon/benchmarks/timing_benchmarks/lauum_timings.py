import functools
import json
import time
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.linalg.lapack import dlauum, slauum

import falkon
from falkon.ooc_ops.ooc_lauum import gpu_lauum
from falkon.utils import devices


def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return out.T
    return out


def run_lauum_exp(exp_name, fn, exp_sizes, num_reps, is_torch, dtype):
    timings = []
    for num_pts in exp_sizes:
        A = gen_random(num_pts, num_pts, dtype=dtype, F=True, seed=123)

        rep_times = []
        for j in range(num_reps):
            if is_torch:
                Ac = torch.from_numpy(A.copy(order="C"))
            else:
                Ac = A.copy(order="F")
            t_s = time.time()
            fn(Ac)
            t_e = time.time()
            rep_times.append(t_e - t_s)
            print(f"Exp {exp_name} - N {num_pts} - Rep {j} - {rep_times[-1]:.2f}s", flush=True)
            del Ac
            if is_torch:
                torch.cuda.empty_cache()
        timings.append(min(rep_times))
    return timings


if __name__ == "__main__":
    init_opt = falkon.FalkonOptions(compute_arch_speed=False)
    gpu_info = [v for k, v in devices.get_device_info(init_opt).items() if k >= 0]
    num_gpu = len(gpu_info)

    experiments: List[Dict[str, Any]] = [
        {
            "exp_name": "OOC 32",
            "exp_sizes": [10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000, 120_000, 140_000],
            "dtype": np.float32,
            "num_reps": 5,
            "is_torch": True,
            "fn": functools.partial(
                gpu_lauum,
                upper=False,
                overwrite=True,
                write_opposite=True,
                opt=falkon.FalkonOptions(compute_arch_speed=False),
            ),
        },
        {
            "exp_name": "OOC 64",
            "exp_sizes": [10_000, 20_000, 30_000, 40_000, 50_000],
            "dtype": np.float64,
            "num_reps": 5,
            "is_torch": True,
            "fn": functools.partial(
                gpu_lauum,
                upper=False,
                overwrite=True,
                write_opposite=True,
                opt=falkon.FalkonOptions(compute_arch_speed=False),
            ),
        },
        {
            "exp_name": "CPU 32",
            "exp_sizes": [10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000],
            "dtype": np.float32,
            "num_reps": 3,
            "is_torch": False,
            "fn": functools.partial(slauum, lower=1, overwrite_c=True),
        },
        {
            "exp_name": "CPU 64",
            "exp_sizes": [10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000],
            "dtype": np.float64,
            "num_reps": 3,
            "is_torch": False,
            "fn": functools.partial(dlauum, lower=1, overwrite_c=True),
        },
    ]
    for exp in experiments:
        exp_times = run_lauum_exp(**exp)
        exp["timings"] = exp_times
    with open(f"logs/lauum_timings_{num_gpu}GPU.json", "w") as fh:
        json.dump(experiments, fh)

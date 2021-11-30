import sys
sys.path.append("..")
import functools

import time
import json

import torch
import numpy as np

import falkon
from falkon.ooc_ops.ooc_potrf import gpu_cholesky
from falkon.la_helpers.cyblas import add_symmetrize
from falkon.utils import devices

DO_RUN = True
RUN_CPU = False

def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return out.T
    return out

def gen_random_pd(t, dtype, F=False, seed=0):
    A = gen_random(t, t, dtype, F, seed)
    add_symmetrize(A)
    #A += A.T
    #A *= 0.5
    A.flat[::t+1] += t
    return A

def run_experiments(experiments):
    for exp in experiments:
        fn = exp['fn']
        for N in exp['N']:
            A = gen_random_pd(N, exp['dt'], F=True, seed=192)

            timings = []
            for j in range(exp['repetitions']):
                if exp['torch']:
                    Ac = torch.from_numpy(A.copy(order='F'))
                else:
                    Ac = A.copy(order='F')
                t_s = time.time()
                fn(Ac)
                t_e = time.time()
                timings.append(t_e - t_s)
                print("Exp %s - N %d - Rep %d - %.2fs" % (exp, N, j, timings[-1]), flush=True)
                if exp['torch']:
                    torch.cuda.empty_cache()
            exp['timings'].append(min(timings))
    return experiments


if __name__ == "__main__":
    init_opt = falkon.FalkonOptions()
    torch.cuda.init()
    gpu_info = [v for k, v in devices.get_device_info(init_opt).items() if k >= 0]
    num_gpu = len(gpu_info)

    defaultN32 = [10_000, 20_000, 30_000, 40_000, 50_000, 65_000, 80_000, 100_000, 120_000, 140_000]
    #defaultN64 = [10_000, 20_000, 30_000, 40_000, 50_000, 65_000, 80_000]
    falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2, compute_arch_speed=False)
    experiments = [
        {
            'name': 'Parallel 32',
            'N': defaultN32,
            'dt': np.float32,
            'timings': [],
            'repetitions': 3,
            'torch': True,
            'fn': functools.partial(gpu_cholesky, upper=False, clean=False, overwrite=True,
                opt=falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2)),
        },
        #{
        #    'name': 'Parallel 64',
        #    'N': defaultN64,
        #    'dt': np.float64,
        #    'timings': [],
        #    'repetitions': 3,
        #    'torch': True,
        #    'fn': functools.partial(gpu_cholesky, upper=False, clean=False, overwrite=True,
        #        opt=falkon.FalkonOptions(chol_force_ooc=True, chol_par_blk_multiplier=2, compute_arch_speed=False)),
        #},
    ]
    if False:
        experiments.extend([
            {
                'name': 'CPU 32',
                'N': defaultN32,
                'dt': np.float32,
                'timings': [],
                'repetitions': 3,
                'torch': False,
                'fn': functools.partial(spotrf, lower=True, clean=False, overwrite_a=True),
            },
            {
                'name': 'CPU 64',
                'N': defaultN64,
                'dt': np.float64,
                'timings': [],
                'repetitions': 2,
                'torch': False,
                'fn': functools.partial(dpotrf, lower=True, clean=False, overwrite_a=True),
            },
        ])

    if DO_RUN:
        timings = run_experiments(experiments)
        for t in timings:
            t['fn'] = str(t['fn'])
            t['dt'] = str(t['dt'])
        with open("logs/potrf_timings_%dGPU.json" % (num_gpu), "w") as fh:
            json.dump(timings, fh)
    else:
        with open("logs/potrf_timings_%dGPU.json" % (num_gpu), "r") as fh:
            timings = json.load(fh)

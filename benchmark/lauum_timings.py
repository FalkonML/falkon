import sys
sys.path.append("..")
import functools

import time
import json

import torch
import numpy as np
from scipy.linalg.lapack import slauum, dlauum

import falkon
from falkon.ooc_ops.ooc_lauum import gpu_lauum
from falkon.utils import devices
from falkon.cuda import initialization

DO_RUN = True
RUN_CPU = False

def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return out.T
    return out

def run_experiments(experiments):
    for exp in experiments:
        fn = exp['fn']
        for N in exp['N']:
            A = gen_random(N, N, exp['dt'], F=True, seed=192)

            timings = []
            for j in range(exp['repetitions']):
                if exp['torch']:
                    Ac = torch.from_numpy(A.copy(order='C'))
                else:
                    Ac = A.copy(order='F')
                t_s = time.time()
                fn(Ac)
                t_e = time.time()
                timings.append(t_e - t_s)
                print("Exp %s - N %d - Rep %d - %.2fs" % (exp, N, j, timings[-1]), flush=True)
                del Ac
                if exp['torch']:
                    torch.cuda.empty_cache()
            exp['timings'].append(min(timings))
    return experiments


if __name__ == "__main__":
    init_opt = falkon.FalkonOptions(compute_arch_speed=False)
    initialization.init(init_opt)
    gpu_info = [v for k, v in devices.get_device_info(init_opt).items() if k >= 0]
    num_gpu = len(gpu_info)
    RUN_CPU = False

    defaultN32 = [10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000, 120_000, 140_000]
    #defaultN64 = [10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000]

    experiments = [
        {
            'name': 'OOC 32',
            'N': [10_000, 20_000, 30_000],
            'dt': np.float32,
            'timings': [],
            'repetitions': 5,
            'torch': True,
            'fn': functools.partial(gpu_lauum, upper=False, overwrite=True, write_opposite=True,
                opt=falkon.FalkonOptions(compute_arch_speed=False)),
        },
        {
            'name': 'OOC 32',
            'N': [40_000, 50_000, 75_000],
            'dt': np.float32,
            'timings': [],
            'repetitions': 5,
            'torch': True,
            'fn': functools.partial(gpu_lauum, upper=False, overwrite=True, write_opposite=True,
                opt=falkon.FalkonOptions(compute_arch_speed=False)),
        },
        {
            'name': 'OOC 32',
            'N': [100_000, 120_000, 140_000],
            'dt': np.float32,
            'timings': [],
            'repetitions': 3,
            'torch': True,
            'fn': functools.partial(gpu_lauum, upper=False, overwrite=True, write_opposite=True,
                opt=falkon.FalkonOptions( compute_arch_speed=False)),
        },
    ]
    #    {
    #        'name': 'OOC 64',
    #        'N': defaultN64,
    #        'dt': np.float64,
    #        'timings': [],
    #        'repetitions': 5,
    #        'torch': True,
    #        'fn': functools.partial(gpu_lauum, upper=False, overwrite=True, write_opposite=True,
    #            opt=falkon.FalkonOptions(compute_arch_speed=False)),
    #    },
    #]
    if RUN_CPU:
        experiments.extend([
            {
                'name': 'CPU 32',
                'N': defaultN32,
                'dt': np.float32,
                'timings': [],
                'repetitions': 3,
                'torch': False,
                'fn': functools.partial(slauum, lower=1, overwrite_c=True),
            },
            #{
            #    'name': 'CPU 64',
            #    'N': defaultN64,
            #    'dt': np.float64,
            #    'timings': [],
            #    'repetitions': 2,
            #    'torch': False,
            #    'fn': functools.partial(dlauum, lower=1, overwrite_c=True),
            #},
        ])

    if DO_RUN:
        timings = run_experiments(experiments)
        for t in timings:
            t['fn'] = str(t['fn'])
            t['dt'] = str(t['dt'])
        with open("logs/lauum_timings_%dGPU.json" % (num_gpu), "w") as fh:
            json.dump(timings, fh)
    else:
        with open("logs/lauum_timings_%dGPU.json" % (num_gpu), "r") as fh:
            timings = json.load(fh)

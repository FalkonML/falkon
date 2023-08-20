import timeit
import json
import argparse
from typing import List, Dict, Any

import torch

from falkon.kernels import GaussianKernel


def gen_data(N, M, D, T, cuda=False, dtype=torch.float64):
    X1 = torch.randn(N, D, requires_grad=False, dtype=dtype)
    X2 = torch.randn(M, D, requires_grad=False, dtype=dtype)
    v = torch.randn(M, T, requires_grad=False, dtype=dtype)
    if not cuda:
        return X1, X2, v
    return X1.cuda(), X2.cuda(), v.cuda()


def run_mmv_exp(exp_name, fn, changing_var, data_sizes, kernel, dtype, num_reps):
    timings = []
    fn = fn + '; torch.cuda.synchronize();'

    for i in range(len(data_sizes[changing_var])):
        N = data_sizes['N'][i] if changing_var == 'N' else data_sizes['N']
        D = data_sizes['D'][i] if changing_var == 'D' else data_sizes['D']
        M = data_sizes['M'][i] if changing_var == 'M' else data_sizes['M']
        T = data_sizes['T'][i] if changing_var == 'T' else data_sizes['T']

        X1, X2, v = gen_data(N, M, D, T, dtype=dtype)

        _vars = locals()
        _vars.update(globals())
        exp_times = timeit.repeat(fn, globals=_vars, number=1, repeat=num_reps)
        timings.append(min(exp_times))
        print("Exp %s - N %d, D %d, M %d, T %d - %.2fs" % (exp_name, N, D, M, T, timings[-1]), flush=True)
        torch.cuda.empty_cache()
    return timings


if __name__ == "__main__":
    aparse = argparse.ArgumentParser(description="MMV experiment runner")
    aparse.add_argument('--num-gpus', type=int, required=True)
    args = aparse.parse_args()
    num_gpus = args.num_gpus

    kernel = GaussianKernel(3.0)
    Ns = [
        1000, 5000, 20000, 50000, 100000, 200000, 400000, 600_000, 1_000_000,
        2_000_000, 10_000_000, 50_000_000, 100_000_000
    ]
    KeopsDs = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    OurDs = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000]
    defaultM = 20_000
    defaultN = 20_000
    defaultT = 10
    defaultD = 10

    experiments: List[Dict[str, Any]] = [
        {
            'exp_name': 'varying N - KeOps 32',
            'changing_var': 'N',
            'data_sizes': {
                'N': Ns,
                'M': defaultM,
                'D': defaultD,
                'T': defaultT,
            },
            'kernel': kernel,
            'dtype': torch.float32,
            'num_reps': 10,
            'fn': ('kernel._keops_mmv_impl(X1, X2, v, kernel, out=None, '
                   'opt=FalkonOptions(keops_active="force", compute_arch_speed=False));'),
        },
        {
            'exp_name': 'varying N - Our 32',
            'changing_var': 'N',
            'data_sizes': {
                'N': Ns,
                'M': defaultM,
                'D': defaultD,
                'T': defaultT,
            },
            'kernel': kernel,
            'dtype': torch.float32,
            'num_reps': 10,
            'fn': ('kernel.mmv(X1, X2, v, out=None, '
                   'opt=FalkonOptions(keops_active="no", compute_arch_speed=False));'),
        },
        {
            'exp_name': 'varying D - KeOps 32',
            'changing_var': 'D',
            'data_sizes': {
                'N': defaultN,
                'M': defaultM,
                'D': KeopsDs,
                'T': defaultT,
            },
            'kernel': kernel,
            'dtype': torch.float32,
            'num_reps': 10,
            'fn': ('kernel.keops_mmv_impl(X1, X2, v, kernel, out=None, '
                   'opt=FalkonOptions(keops_active="force", compute_arch_speed=False));'),
        },
        {
            'exp_name': 'varying D - Our 32',
            'changing_var': 'D',
            'data_sizes': {
                'N': defaultN,
                'M': defaultM,
                'D': OurDs,
                'T': defaultT,
            },
            'kernel': kernel,
            'dtype': torch.float32,
            'num_reps': 10,
            'fn': 'kernel.mmv(X1, X2, v, out=None, opt=FalkonOptions(keops_active="no", compute_arch_speed=False));'
        },
    ]

    for exp in experiments:
        exp_times = run_mmv_exp(**exp)
        exp['timings'] = exp_times
        # Remove the stuff we can't serialize
        exp['kernel'] = None
        exp['dtype'] = None
    with open("logs/mmv_timings_%dGPU.json" % (num_gpus), "w") as fh:
        json.dump(experiments, fh)

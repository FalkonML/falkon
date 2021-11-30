import sys
sys.path.append("..")

import timeit
import json
import argparse

import torch

from falkon.kernels import GaussianKernel


def gen_data(N, M, D, T, cuda=False, dtype=torch.float64):
    X1 = torch.randn(N, D, requires_grad=False, dtype=dtype)
    X2 = torch.randn(M, D, requires_grad=False, dtype=dtype)
    v = torch.randn(M, T, requires_grad=False, dtype=dtype)
    if not cuda:
        return X1, X2, v
    return X1.cuda(), X2.cuda(), v.cuda()


def run_experiments(experiments):
    for exp in experiments:
        vname = exp['variable']
        variable = exp[vname]
        for i in range(len(variable)):
            N = exp['N'][i] if vname == 'N' else exp['N']
            D = exp['D'][i] if vname == 'D' else exp['D']
            M = exp['M'][i] if vname == 'M' else exp['M']
            T = exp['T'][i] if vname == 'T' else exp['T']
            X1, X2, v = gen_data(N, M, D, T, dtype=exp['dt'])
            kernel = exp['kernel']
            fn = exp['fn'] + ' torch.cuda.synchronize()'
            _vars = locals()
            _vars.update(globals())
            timings = timeit.repeat(fn, globals=_vars, number=1, repeat=exp['repetitions'])
            exp['timings'].append(min(timings))
            print(exp, flush=True)
            torch.cuda.empty_cache()
    return experiments


if __name__ == "__main__":
    aparse = argparse.ArgumentParser(description="MMV experiment runner")
    aparse.add_argument('--num-gpus', type=int, required=True)
    args = aparse.parse_args()
    num_gpus = args.num_gpus

    kernel = GaussianKernel(3.0)
    Ns = [1000, 5000, 20000, 50000, 100000, 200000, 400000, 600_000, 1_000_000, 2_000_000, 10_000_000, 50_000_000, 100_000_000]
    KeopsDs = [10, 50, 100, 250, 500, 750, 1000, 1250]
    OurDs = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000]
    defaultM = 20_000
    defaultN = 20_000
    defaultT = 10
    defaultD = 10

    experiments = [
        {
            'name': 'varying N - KeOps 32',
            'variable': 'N',
            'N': Ns,
            'M': defaultM,
            'D': defaultD,
            'T': defaultT,
            'kernel': kernel,
            'dt': torch.float32,
            'timings': [],
            'repetitions': 10,
            'fn': 'kernel._keops_mmv_impl(X1, X2, v, kernel, out=None, opt=FalkonOptions(keops_active="force", compute_arch_speed=False));',
        },
        {
            'name': 'varying N - Our 32',
            'variable': 'N',
            'N': Ns,
            'M': defaultM,
            'D': defaultD,
            'T': defaultT,
            'kernel': kernel,
            'dt': torch.float32,
            'timings': [],
            'repetitions': 10,
            'fn': 'kernel.mmv(X1, X2, v, out=None, opt=FalkonOptions(keops_active="no", compute_arch_speed=False));'
        },
        {
            'name': 'varying D - KeOps 32',
            'variable': 'D',
            'N': defaultN,
            'M': defaultM,
            'D': [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
            'T': defaultT,
            'kernel': kernel,
            'dt': torch.float32,
            'timings': [],
            'repetitions': 10,
            'fn': 'kernel._keops_mmv_impl(X1, X2, v, kernel, out=None, opt=FalkonOptions(keops_active="force", compute_arch_speed=False));'
        },
        {
            'name': 'varying D - Our 32',
            'variable': 'D',
            'N': defaultN,
            'M': defaultM,
            'D': OurDs,
            'T': defaultT,
            'kernel': kernel,
            'dt': torch.float32,
            'timings': [],
            'repetitions': 20,
            'fn': 'kernel.mmv(X1, X2, v, out=None, opt=FalkonOptions(keops_active="no", compute_arch_speed=False));'
        },
    ]

    timings = run_experiments(experiments)
    for t in timings:  # Remove the stuff we can't serialize
        t['kernel'] = None
        t['dt'] = None
    with open("logs/mmv_timings_%dGPU.json" % (num_gpus), "w") as fh:
        json.dump(timings, fh)


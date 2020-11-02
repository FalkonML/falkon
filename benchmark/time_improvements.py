"""
Time Falkon with different parts switched on/off on a sample dataset (MS) which is quite fast:

Baseline (equivalent to FALKON MATLAB):
1. float64 + CPU Preconditioner + single GPU (no keops)
2. float32 + CPU Preconditioner + single GPU (no keops)
3. float32 + GPU Preconditioner + single GPU (no keops)
4. float32 + GPU Preconditioner + 2 GPU (no keops)
5. float32 + GPU Preconditioner + 2 GPU (keops)
"""
import argparse
import dataclasses
import functools
import time

import numpy as np
import torch

from benchmark_utils import DataType, Dataset
from datasets import get_load_fn
from error_metrics import get_err_fns
import falkon
from falkon import kernels
from falkon.cuda import initialization

RANDOM_SEED = 95

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def run(exp_num, dset, show_intermediate_errors: bool = False):
    opt = falkon.FalkonOptions(
        chol_par_blk_multiplier=2, debug=True,
        pc_epsilon_32=1e-6, pc_epsilon_64=1e-13,
        compute_arch_speed=False,
        num_fmm_streams=2, no_single_kernel=False)
    params = {
        'seed': 12,
        'kernel': kernels.GaussianKernel(3.8),
        'penalty': 1e-7,
        'M': 100_000,
        'maxiter': 10,
    }
    if exp_num == 1:
        opt = dataclasses.replace(opt, cpu_preconditioner=True, keops_active="no")
        dtype = DataType.float64
    elif exp_num == 2:
        opt = dataclasses.replace(opt, cpu_preconditioner=True, keops_active="no")
        dtype = DataType.float32
    elif exp_num == 3:
        opt = dataclasses.replace(opt, cpu_preconditioner=False, keops_active="no")
        dtype = DataType.float32
    elif exp_num == 4:
        opt = dataclasses.replace(opt, cpu_preconditioner=False, keops_active="no")
        dtype = DataType.float32
    elif exp_num == 5:
        opt = dataclasses.replace(opt, cpu_preconditioner=False, keops_active="force")
        dtype = DataType.float32
    else:
        raise ValueError("exp num %d not valid" % (exp_num))
    data = load_data(dset, data_type=dtype)
    torch.cuda.init()
    initialization.init(opt)
    print("\n\n --- Running Experiment %d -- %s" % (exp_num, opt))
    data = list(data)
    data[0] = data[0].pin_memory()
    data[1] = data[1].pin_memory()
    data[2] = data[2].pin_memory()
    data[3] = data[3].pin_memory()
    t_s = time.time()
    flk = run_single(dset, data[0], data[1], data[2], data[3], data[4], show_intermediate_errors, opt, params)
    t_e = time.time()
    print("Timing for Experiment %d: %s -- fit times %s" % (exp_num, t_e - t_s, flk.fit_times_))


def load_data(dset, data_type):
    load_fn = get_load_fn(dset)
    return load_fn(dtype=data_type.to_numpy_dtype(), as_torch=True)


def run_single(dset, Xtr, Ytr, Xts, Yts, kwargs, intermediate_errors, opt, params):
    err_fns = get_err_fns(dset)
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
    error_every = 1 if intermediate_errors else None

    flk = falkon.Falkon(
        error_fn=err_fns[0],
        error_every=error_every,
        options=opt,
        **params
    )
    flk.fit(Xtr, Ytr, Xts, Yts)
    return flk


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")
    p.add_argument('-i', '--exp-num', type=int, required=True,
                   help='The experiment type, 1 to 5.')
    p.add_argument('-d', '--dataset', type=Dataset, choices=list(Dataset), required=True,
                   help='Dataset')
    args = p.parse_args()
    run(args.exp_num, args.dataset, show_intermediate_errors=True)

import time

import kernels
import numpy as np
import torch

import falkon


def gen_random(a, b, dtype, F=False, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.random(size=(a, b), dtype=dtype)
    if F:
        return out.T
    return out


def run_mm_exp(exp_name, kernel, N, D, pin_memory, num_reps):
    timings = []
    A = torch.randn(N, D, dtype=torch.float32)
    if pin_memory:
        A = A.pin_memory()

    for _ in range(num_reps):
        t_s = time.time()
        kernel(A)
        torch.cuda.synchronize()
        t_e = time.time()
        timings.append(t_e - t_s)
        print(f"{exp_name} - {N=} {D=} {pin_memory=} - {t_e - t_s:.2f}s", flush=True)
    print(f"\t min={np.min(timings):.2f}s")
    return np.min(timings)


if __name__ == "__main__":
    N = 50_000
    D = 256
    for no_single_kernel in [True, False]:
        init_opt = falkon.FalkonOptions(compute_arch_speed=False, no_single_kernel=no_single_kernel)
        kernel = kernels.GaussianKernel(sigma=5.0, opt=init_opt)
        exp_name = f"exp-{no_single_kernel=}"
        run_mm_exp(exp_name=exp_name, kernel=kernel, N=N, D=D, pin_memory=True, num_reps=5)
        print()

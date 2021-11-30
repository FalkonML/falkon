import dataclasses
import time

import numpy as np
import pytest
from falkon import FalkonOptions
from pytest import mark
import torch

from falkon.tests.conftest import fix_mat, memory_checker
from falkon.utils import decide_cuda
from falkon.mmv_ops.kernel_trsm import kernel_trsm_fro
from falkon.kernels import GaussianKernel
from falkon.tests.gen_random import gen_random


n, m, d = 1000, 100, 10
max_mem = 0.2 * 2**20
rtol = {
    np.float32: 1e-5,
    np.float64: 1e-13
}
cuda_skip = mark.skipif(not decide_cuda(), reason="No GPU found.")


@pytest.fixture(scope="module")
def m1() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def m2() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def kernel():
    return GaussianKernel(sigma=1.0)


@pytest.fixture(scope="module")
def L(m2, kernel):
    kmm = kernel(m2, m2) + torch.eye(m2.shape[0], dtype=m2.dtype, device=m2.device) * 1e-5
    return torch.linalg.cholesky(kmm)


@pytest.mark.parametrize("m1_order", ['F', 'C'])
@pytest.mark.parametrize("m2_order", ['F', 'C'])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transpose", [True, False], ids=['trans', 'notrans'])
class TestKernelTrsmFro():
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="no",
                                  max_gpu_mem=max_mem, max_cpu_mem=max_mem)

    @staticmethod
    def expected_kernel_trsm_fro(tri, m1, m2, kernel, lower, transpose):
        kmn = kernel(m2, m1)
        solve = torch.triangular_solve(kmn, tri, upper=not lower, transpose=transpose).solution
        return solve.sum().item()

    @pytest.mark.parametrize("comp_dev,data_dev", [
        ("cpu", "cpu"),
        pytest.param("cuda", "cpu", marks=[cuda_skip]),
        pytest.param("cuda", "cuda", marks=[cuda_skip])
    ])
    def test(self, m1, m2, kernel, L, comp_dev, data_dev, m1_order, m2_order, dtype, transpose):
        opt = dataclasses.replace(self.basic_options, use_cpu=comp_dev == 'cpu')
        m1 = fix_mat(m1, order=m1_order, dtype=dtype, device=data_dev)
        m2 = fix_mat(m2, order=m2_order, dtype=dtype, device=data_dev)
        L = fix_mat(L, order='F', dtype=dtype, device=data_dev)

        with memory_checker(opt) as new_opt:
            out = kernel_trsm_fro(
                tri=L, m1=m1, m2=m2, kernel=kernel, lower=True, transpose=transpose, opt=new_opt)
        exp = TestKernelTrsmFro.expected_kernel_trsm_fro(
            L, m1, m2, kernel, lower=True, transpose=transpose)

        np.testing.assert_allclose(out, exp, rtol=rtol[dtype])


@pytest.mark.parametrize("comp_dev,data_dev", [
    #("cpu", "cpu"),
    pytest.param("cuda", "cpu", marks=[cuda_skip]),
    pytest.param("cuda", "cuda", marks=[cuda_skip])
])
@pytest.mark.benchmark
def test_kernel_trsm_fro_perf(kernel, comp_dev, data_dev):
    opt = FalkonOptions(use_cpu=comp_dev == 'cpu')
    n, m, d = 200_000, 5000, 10
    num_rep = 5

    m1 = torch.from_numpy(gen_random(n, d, 'float32', F=False, seed=92)).to(device=data_dev)
    m2 = torch.from_numpy(gen_random(m, d, 'float32', F=False, seed=92)).to(device=data_dev)
    kmm = kernel(m2, m2) + torch.eye(m2.shape[0], dtype=m2.dtype, device=m2.device) * 1e-3
    L = torch.linalg.cholesky(kmm)

    times = []
    for i in range(num_rep):
        t_s = time.time()
        kernel_trsm_fro(tri=L, m1=m1, m2=m2, kernel=kernel, lower=True, transpose=False, opt=opt)
        times.append(time.time() - t_s)

    print(f"Kernel-TRSM-Fro: computations on {comp_dev}, data on {data_dev}, n={n}, m={m}: {min(times):.3f}s")

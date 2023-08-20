import dataclasses
import unittest

import numpy as np
import pytest
import torch

from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.preconditioner import FalkonPreconditioner
from falkon.tests.conftest import fix_mat
from falkon.tests.gen_random import gen_random
from falkon.utils import decide_cuda
from falkon.utils.tensor_helpers import move_tensor


def assert_invariant_on_TT(prec, kMM, tol=1e-8):
    """
    T = chol(kMM) => T.T @ T = kMM
    If we solve T.T @ x = kMM we should get that x = T
    """
    T = prec.invTt(kMM)
    assert T.dtype == kMM.dtype, "Wrong data-type"

    np.testing.assert_allclose((T.T @ T).cpu().numpy(), kMM.cpu().numpy(), rtol=tol, atol=tol)


def assert_invariant_on_AT(prec, kMM, la, tol=1e-8):
    """
    T = chol(kMM) => T.T @ T = kMM
    A = chol(1/M T@T.T + la*I) => A.T @ A = ...
    We show we can recover first T and then A
    """
    M = kMM.shape[0]
    T = prec.invTt(kMM)
    ATA = (1.0 / M) * T @ T.T + la * torch.eye(M, dtype=kMM.dtype, device=kMM.device)
    A = prec.invAt(ATA)
    assert A.dtype == kMM.dtype, "Wrong data-type"

    np.testing.assert_allclose((A.T @ A).cpu().numpy(), ATA.cpu().numpy(), rtol=tol, atol=tol)


def assert_invariant_on_T(prec, kMM, tol=1e-8):
    """
    We should recover that T.T @ T = kMM
    """
    M = kMM.shape[0]
    out = prec.invT(prec.invTt(kMM))
    assert out.dtype == kMM.dtype, "Wrong data-type"

    out = out.cpu().numpy()
    np.testing.assert_allclose(out, np.eye(M, dtype=out.dtype), rtol=tol, atol=tol)


def assert_invariant_on_prec(prec, n, kMM, la, tol=1e-8):
    """
    The preconditioner 1/n * B@B.T should be = (n/M kMM.T @ kMM + la kMM)^-1
    We assert that (B@B.T)^-1 = (n/M kMM.T @ kMM + la kMM)
    """
    M = kMM.shape[0]
    desiredPrec = (n / M) * kMM.T @ kMM + la * n * kMM

    out = (1 / n) * prec.invT(prec.invA(prec.invAt(prec.invTt(desiredPrec))))
    assert out.dtype == desiredPrec.dtype, "Wrong data-type"
    out = out.cpu().numpy()
    np.testing.assert_allclose(out, np.eye(M, dtype=out.dtype), rtol=tol, atol=tol)


M = 100
N = 50_000  # Used only for normalization.


@pytest.fixture()
def rtol():
    return {np.float64: 1e-10, np.float32: 1e-2}


@pytest.fixture(scope="module")
def kernel():
    return GaussianKernel(10.0)


@pytest.fixture(scope="module")
def mat():
    return gen_random(M, M, "float64", F=True, seed=10)


@pytest.fixture(scope="module")
def gram(kernel, mat):
    opt = FalkonOptions(compute_arch_speed=False, no_single_kernel=True, use_cpu=True)
    return kernel(torch.from_numpy(mat), torch.from_numpy(mat), opt=opt)


@pytest.mark.parametrize(
    "cpu",
    [pytest.param(True), pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])],
    ids=["cpu", "gpu"],
)
class TestFalkonPreconditioner:
    basic_opt = FalkonOptions(compute_arch_speed=False, no_single_kernel=True)

    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
    def test_simple(self, mat, kernel, gram, cpu, dtype, order, rtol):
        opt = dataclasses.replace(self.basic_opt, use_cpu=cpu, cpu_preconditioner=cpu)
        rtol = rtol[dtype]

        mat = fix_mat(mat, dtype=dtype, order=order, copy=True)
        gram = fix_mat(gram, dtype=dtype, order=order, copy=True)

        la = 100
        prec = FalkonPreconditioner(la, kernel, opt)
        prec.init(mat)
        assert_invariant_on_TT(prec, gram, tol=rtol)
        assert_invariant_on_AT(prec, gram, la, tol=rtol)
        assert_invariant_on_T(prec, gram, tol=rtol * 10)
        assert_invariant_on_prec(prec, N, gram, la, tol=rtol * 10)

    def test_zero_lambda(self, mat, kernel, gram, cpu, rtol):
        opt = dataclasses.replace(self.basic_opt, use_cpu=cpu, cpu_preconditioner=cpu)
        mat = fix_mat(mat, dtype=np.float64, order="K", copy=True)
        gram = fix_mat(gram, dtype=np.float64, order="K", copy=True)

        la = 0
        prec = FalkonPreconditioner(la, kernel, opt)
        prec.init(mat)
        assert_invariant_on_TT(prec, gram, tol=1e-10)
        assert_invariant_on_AT(prec, gram, la, tol=1e-10)
        assert_invariant_on_T(prec, gram, tol=1e-9)
        assert_invariant_on_prec(prec, N, gram, la, tol=1e-8)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
def test_incore(mat, kernel, gram, dtype, order, rtol):
    opt = FalkonOptions(no_single_kernel=True, use_cpu=False, cpu_preconditioner=False)
    _rtol = rtol[dtype]

    mat = fix_mat(mat, dtype=dtype, order=order, copy=True)
    gpu_mat = move_tensor(mat, "cuda:0")
    gram = fix_mat(gram, dtype=dtype, order=order, copy=True)
    gpu_gram = move_tensor(gram, "cuda:0")

    la = 1

    prec = FalkonPreconditioner(la, kernel, opt)
    prec.init(mat)

    gpu_prec = FalkonPreconditioner(la, kernel, opt)
    gpu_prec.init(gpu_mat)

    np.testing.assert_allclose(prec.dT.numpy(), gpu_prec.dT.cpu().numpy(), rtol=_rtol)
    np.testing.assert_allclose(prec.dA.numpy(), gpu_prec.dA.cpu().numpy(), rtol=_rtol)
    np.testing.assert_allclose(prec.fC.numpy(), gpu_prec.fC.cpu().numpy(), rtol=_rtol * 10)
    assert gpu_prec.fC.device == gpu_mat.device, "Device changed unexpectedly"

    assert_invariant_on_TT(gpu_prec, gpu_gram, tol=_rtol)
    assert_invariant_on_AT(prec, gram, la, tol=_rtol)
    assert_invariant_on_T(prec, gram, tol=_rtol * 10)
    assert_invariant_on_prec(prec, N, gram, la, tol=_rtol * 10)


@pytest.mark.full
@unittest.skipIf(not decide_cuda(), "No GPU found.")
def test_cpu_gpu_equality(mat, kernel, gram):
    la = 12.3

    mat = fix_mat(mat, dtype=np.float64, order="F", copy=True)

    opt = FalkonOptions(compute_arch_speed=False, use_cpu=False, cpu_preconditioner=False)
    prec_gpu = FalkonPreconditioner(la, kernel, opt)
    prec_gpu.init(mat)

    opt = dataclasses.replace(opt, use_cpu=True, cpu_preconditioner=True)
    prec_cpu = FalkonPreconditioner(la, kernel, opt)
    prec_cpu.init(mat)

    np.testing.assert_allclose(prec_cpu.fC, prec_gpu.fC, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(prec_cpu.dA, prec_gpu.dA, rtol=1e-10)
    np.testing.assert_allclose(prec_cpu.dT, prec_gpu.dT, rtol=1e-10)

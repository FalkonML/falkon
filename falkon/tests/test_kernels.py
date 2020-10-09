import abc
import dataclasses

import numpy as np
import pytest
import torch

from falkon.kernels import *
from falkon.options import FalkonOptions
from falkon.tests.conftest import memory_checker
from falkon.tests.naive_kernels import *
from falkon.tests.gen_random import gen_random
from falkon.utils import decide_cuda
from falkon.utils.switches import decide_keops


# Global data dimensions
n = 2000
m = 1500
d = 10
t = 5


def _run_test(fn, exp, tensors, out, rtol, opt):
    with memory_checker(opt) as new_opt:
        actual = fn(*tensors, out=out, opt=new_opt)

    # Check 1. Accuracy
    np.testing.assert_allclose(exp, actual, rtol=rtol)
    # Check 2. Output pointers
    if out is not None:
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


@pytest.fixture(scope="module")
def A() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def B() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def v() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, t, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def w() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, t, 'float64', False, seed=92))


@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
], ids=["cpu", "gpu"])
class AbstractKernelTester(abc.ABC):
    max_mem = 2 * 2**20
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False,
                                  max_cpu_mem=max_mem, max_gpu_mem=max_mem)
    _RTOL = {
        torch.float32: 1e-5,
        torch.float64: 1e-12
    }

    @pytest.fixture(scope="class")
    def exp_v(self, exp_k: np.ndarray, v: torch.Tensor) -> np.ndarray:
        return exp_k @ v.numpy()

    @pytest.fixture(scope="class")
    def exp_dv(self, exp_k: np.ndarray, v: torch.Tensor) -> np.ndarray:
        return exp_k.T @ (exp_k @ v.numpy())

    @pytest.fixture(scope="class")
    def exp_dw(self, exp_k: np.ndarray, w: torch.Tensor) -> np.ndarray:
        return exp_k.T @ w.numpy()

    @pytest.fixture(scope="class")
    def exp_dvw(self, exp_k: np.ndarray, v: torch.Tensor, w: torch.Tensor) -> np.ndarray:
        return exp_k.T @ (exp_k @ v.numpy() + w.numpy())

    def test_kernel(self, kernel, A, B, exp_k, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        _run_test(kernel, exp_k, (A, B), out=None, rtol=self._RTOL[A.dtype], opt=opt)

    @pytest.mark.parametrize("keops", [
        pytest.param("force", marks=pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")),
        "no"
    ], ids=["KeOps", "No KeOps"])
    def test_mmv(self, kernel, keops, A, B, v, exp_v, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, keops_active=keops)
        _run_test(kernel.mmv, exp_v, (A, B, v), out=None, rtol=self._RTOL[A.dtype], opt=opt)

    @pytest.mark.parametrize("keops", [
        pytest.param("force", marks=pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")),
        "no"
    ], ids=["KeOps", "No KeOps"])
    def test_dv(self, kernel, keops, A, B, v, exp_dv, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, keops_active=keops)
        _run_test(kernel.dmmv, exp_dv, (A, B, v, None), out=None, rtol=self._RTOL[A.dtype], opt=opt)

    @pytest.mark.parametrize("keops", [
        pytest.param("force", marks=pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")),
        "no"
    ], ids=["KeOps", "No KeOps"])
    def test_dw(self, kernel, keops, A, B, w, exp_dw, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, keops_active=keops)
        _run_test(kernel.dmmv, exp_dw, (A, B, None, w), out=None, rtol=self._RTOL[A.dtype], opt=opt)

    @pytest.mark.parametrize("keops", [
        pytest.param("force", marks=pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")),
        "no"
    ], ids=["KeOps", "No KeOps"])
    def test_dvw(self, kernel, keops, A, B, v, w, exp_dvw, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, keops_active=keops)
        _run_test(kernel.dmmv, exp_dvw, (A, B, v, w), out=None, rtol=self._RTOL[A.dtype], opt=opt)


class TestGaussianKernel(AbstractKernelTester):
    @pytest.fixture(scope="class")
    def single_sigma(self) -> float:
        return 2

    @pytest.fixture(scope="class")
    def vector_sigma(self, single_sigma: float) -> torch.Tensor:
        equiv_sigma = 1 / (single_sigma ** 2)
        return torch.tensor([equiv_sigma] * d, dtype=torch.float64)

    @pytest.fixture(scope="class")
    def mat_sigma(self, vector_sigma: torch.Tensor) -> torch.Tensor:
        return torch.diag(vector_sigma)

    @pytest.fixture(scope="class")
    def exp_k(self, A: torch.Tensor, B: torch.Tensor, single_sigma: float) -> np.ndarray:
        return naive_gaussian_kernel(A.numpy(), B.numpy(), single_sigma)

    @pytest.fixture(params=[1, 2, 3, 4], ids=[
        "single-sigma", "vec-sigma", "vec-sigma-flat", "mat-sigma"],
        scope="class")
    def kernel(self, single_sigma, vector_sigma, mat_sigma, request):
        if request.param == 1:
            return GaussianKernel(single_sigma)
        elif request.param == 2:
            return GaussianKernel(vector_sigma)
        elif request.param == 3:
            return GaussianKernel(vector_sigma.reshape(-1, 1))
        elif request.param == 4:
            return GaussianKernel(mat_sigma)

    def test_wrong_sigma_dims(self, A, B, cpu):
        sigmas = torch.tensor([2.0] * (d - 1), dtype=torch.float64)
        kernel = GaussianKernel(sigma=sigmas)
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu,
                                  max_gpu_mem=np.inf, max_cpu_mem=np.inf)
        with pytest.raises(RuntimeError) as excinfo:
            _run_test(kernel, None, (A, B), out=None, rtol=self._RTOL[A.dtype], opt=opt)

        if cpu:
            assert "size mismatch" in str(excinfo.value)
        # If on GPU the 'size mismatch' message is in the base exception (since it's reraised
        # by PropagatingThread) but I'm not sure how to fetch it.


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
def test_gaussian_pd():
    X = gen_random(10000, 2, 'float32', F=True, seed=12)
    Xt = torch.from_numpy(X)
    sigma = 10.0
    opt = FalkonOptions(compute_arch_speed=False, max_gpu_mem=1 * 2**30, use_cpu=False,
                        no_single_kernel=False)
    k = GaussianKernel(sigma, opt=opt)
    actual = k(Xt, Xt, opt=opt)
    actual += torch.eye(Xt.shape[0]) * (1e-7 * Xt.shape[0])
    # Test positive definite
    np.linalg.cholesky(actual)


class TestLaplacianKernel(AbstractKernelTester):
    _RTOL = {
        torch.float32: 1e-5,
        torch.float64: 4e-8
    }

    @pytest.fixture(scope="class")
    def kernel(self) -> LaplacianKernel:
        return LaplacianKernel(sigma=2)

    @pytest.fixture(scope="class")
    def exp_k(self, A: torch.Tensor, B: torch.Tensor, kernel: LaplacianKernel) -> np.ndarray:
        return naive_laplacian_kernel(A.numpy(), B.numpy(), sigma=kernel.sigma)


class TestLinearKernel(AbstractKernelTester):
    @pytest.fixture(scope="class")
    def kernel(self) -> LinearKernel:
        return LinearKernel(beta=2, sigma=2)

    @pytest.fixture(scope="class")
    def exp_k(self, A: torch.Tensor, B: torch.Tensor, kernel: LinearKernel) -> np.ndarray:
        return naive_linear_kernel(A.numpy(), B.numpy(), beta=kernel.beta, sigma=kernel.sigma)


class TestSigmoidKernel(AbstractKernelTester):
    @pytest.fixture(scope="class")
    def kernel(self) -> SigmoidKernel:
        return SigmoidKernel(alpha=3, beta=2)

    @pytest.fixture(scope="class")
    def exp_k(self, A: torch.Tensor, B: torch.Tensor, kernel: SigmoidKernel) -> np.ndarray:
        return naive_sigmoid_kernel(A.numpy(), B.numpy(), alpha=kernel.alpha.item(), beta=kernel.beta.item())


class TestPolynomialKernel(AbstractKernelTester):
    @pytest.fixture(scope="class", params=[1, 2], ids=["poly1.4", "poly2.0"])
    def kernel(self, request) -> PolynomialKernel:
        if request.param == 1:
            return PolynomialKernel(alpha=2.0, beta=3, degree=1.4)
        elif request.param == 2:
            return PolynomialKernel(alpha=2.0, beta=3, degree=2.0)

    @pytest.fixture(scope="class")
    def exp_k(self, A: torch.Tensor, B: torch.Tensor, kernel: PolynomialKernel) -> np.ndarray:
        return naive_polynomial_kernel(A.numpy(), B.numpy(), alpha=kernel.alpha.item(),
                                       beta=kernel.beta.item(), degree=kernel.degree.item())

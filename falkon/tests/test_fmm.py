import dataclasses

import numpy as np
import pytest
import torch
from falkon.utils.tensor_helpers import move_tensor

from falkon.options import FalkonOptions

from falkon.kernels import GaussianKernel, LinearKernel, PolynomialKernel
from falkon.tests.conftest import memory_checker, numpy_to_torch_type
from falkon.tests.naive_kernels import naive_gaussian_kernel, naive_linear_kernel, naive_polynomial_kernel
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.utils import decide_cuda


# Global data dimensions
n = 2000
m = 1500
d = 10
s_n = 500
s_m = 550
s_d = 20000
density = 1e-4


def _run_fmm_test(k_class, k_exp, A, B, out, dtype, rtol, opt):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A.astype(dtype, copy=False))
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B.astype(dtype, copy=False))
    if out is not None and isinstance(out, np.ndarray):
        out = torch.from_numpy(out.astype(dtype, copy=False))

    with memory_checker(opt) as new_opt:
        actual = k_class(A, B, out=out, opt=new_opt)

    np.testing.assert_allclose(k_exp, actual.cpu().numpy(), rtol=rtol)
    if out is not None:
        # Check output pointers
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


# noinspection PyPep8Naming
@pytest.fixture(scope="module")
def Ac():
    return gen_random(n, d, 'float64', False, seed=92)


# noinspection PyPep8Naming
@pytest.fixture(scope="module")
def Af():
    return np.asfortranarray(gen_random(n, d, 'float64', False, seed=92))


# noinspection PyPep8Naming
@pytest.fixture(scope="module")
def Bc():
    return gen_random(m, d, 'float64', False, seed=92)


# noinspection PyPep8Naming
@pytest.fixture(scope="module")
def Bf():
    return np.asfortranarray(gen_random(m, d, 'float64', False, seed=92))


@pytest.fixture(scope="module", ids=["K-gaussian"])
def k1():
    return GaussianKernel(sigma=1)


@pytest.fixture(scope="module", ids=["K-linear"])
def k2():
    return LinearKernel()


@pytest.fixture(scope="module", ids=["K-polynomial"])
def k3():
    return PolynomialKernel(1.2, 3, 2.5)


@pytest.fixture(scope="module")
def expected1(Af, Bf):
    return naive_gaussian_kernel(Af, Bf, 1)


@pytest.fixture(scope="module")
def expected2(Af, Bf):
    return naive_linear_kernel(Af, Bf, 0, 1)


@pytest.fixture(scope="module")
def expected3(Af, Bf):
    return naive_polynomial_kernel(Af, Bf, 1.2, 3, 2.5)


@pytest.fixture(scope="class")
def s_A():
    A = gen_sparse_matrix(s_n, s_d, np.float64, density=density, seed=14)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="class")
def s_B():
    B = gen_sparse_matrix(s_m, s_d, np.float64, density=density, seed=14)
    Bd = torch.from_numpy(B.to_scipy().todense())
    return B, Bd


@pytest.fixture(scope="class")
def s_expected1(s_A, s_B):
    return naive_gaussian_kernel(s_A[1], s_B[1], 1)


@pytest.fixture(scope="class")
def s_expected2(s_A, s_B):
    return naive_linear_kernel(s_A[1], s_B[1], 0, 1)


@pytest.fixture(scope="class")
def s_expected3(s_A, s_B):
    return naive_polynomial_kernel(s_A[1], s_B[1], 1.2, 3, 2.5)


@pytest.fixture
def k_class(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def k_exp(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def A(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def B(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("k_class,k_exp", [
    pytest.param('k1', 'expected1', marks=pytest.mark.usefixtures('k1', 'expected1')),
    pytest.param('k2', 'expected2', marks=pytest.mark.usefixtures('k2', 'expected2')),
    pytest.param('k3', 'expected3', marks=pytest.mark.usefixtures('k3', 'expected3')),
], indirect=True)
@pytest.mark.parametrize("cpu,input_device", [
    pytest.param(True, "cpu"),
    pytest.param(False, "cpu", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")]),
    pytest.param(False, "cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")]),
], ids=["cpu-cpu_input", "gpu-cpu_input", "gpu-gpu_input"])
class TestDenseFmm:
    max_mem = 2 * 2**20
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, no_single_kernel=False,
                                  max_cpu_mem=max_mem, max_gpu_mem=max_mem)
    _RTOL = {
        torch.float32: 1e-5,
        torch.float64: 1e-12
    }

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("A,B", [
        pytest.param('Ac', 'Bc', marks=pytest.mark.usefixtures('Ac', 'Bc')),
        pytest.param('Af', 'Bf', marks=pytest.mark.usefixtures('Af', 'Bf')),
        pytest.param('Ac', 'Bf', marks=pytest.mark.usefixtures('Ac', 'Bf')),
    ], indirect=True)
    def test(self, A, B, k_class, k_exp, dtype, cpu, input_device):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        if input_device.startswith("cuda"):
            # For fMM there is nothing we can do about CUDA memory usage!
            opt = dataclasses.replace(opt, max_gpu_mem=np.inf)
        A = move_tensor(torch.from_numpy(A), input_device)
        B = move_tensor(torch.from_numpy(B), input_device)

        _run_fmm_test(k_class, k_exp, A, B, out=None, dtype=dtype, opt=opt, rtol=self._RTOL[A.dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_with_out(self, Ac: np.ndarray, Bc: np.ndarray, k_class, k_exp, dtype, cpu, input_device):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        Ac = move_tensor(torch.from_numpy(Ac.astype(dtype)), input_device)
        Bc = move_tensor(torch.from_numpy(Bc.astype(dtype)), input_device)
        out = torch.empty(Ac.shape[0], Bc.shape[0], dtype=Ac.dtype, device=input_device)

        _run_fmm_test(k_class, k_exp, Ac, Bc, out=out, dtype=dtype, opt=opt, rtol=self._RTOL[Ac.dtype])

    @pytest.mark.parametrize("A,B", [
        pytest.param('Af', 'Bf', marks=pytest.mark.usefixtures('Af', 'Bf')),
        pytest.param('Ac', 'Bf', marks=pytest.mark.usefixtures('Ac', 'Bf')),
    ], indirect=True)
    def test_precise_kernel(self, A, B, k_class, k_exp, cpu, input_device):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, no_single_kernel=True)

        A = move_tensor(torch.from_numpy(A), input_device)
        B = move_tensor(torch.from_numpy(B), input_device)
        out = torch.empty(A.shape[0], B.shape[0], dtype=A.dtype, device=input_device)
        # Note rtol is 10x lower than in the other tests
        _run_fmm_test(k_class, k_exp, A, B, out=out, dtype=np.float32, opt=opt, rtol=1e-6)


@pytest.mark.parametrize("k_class,k_exp", [
    pytest.param('k1', 's_expected1', marks=pytest.mark.usefixtures('k1', 's_expected1')),
    pytest.param('k2', 's_expected2', marks=pytest.mark.usefixtures('k2', 's_expected2')),
    pytest.param('k3', 's_expected3', marks=pytest.mark.usefixtures('k3', 's_expected3')),
], indirect=True)
@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
])
class TestSparseFmm:
    max_mem = 50 * 2**20
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, no_single_kernel=True,
                                  max_cpu_mem=max_mem, max_gpu_mem=max_mem)
    _RTOL = {
        torch.float32: 1e-5,
        torch.float64: 1e-12
    }

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sparse(self, k_class, k_exp, s_A, s_B, dtype, cpu):
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)

        A_sparse = s_A[0].to(dtype=numpy_to_torch_type(dtype))
        B_sparse = s_B[0].to(dtype=numpy_to_torch_type(dtype))

        # Here both A and B are sparse
        _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=None, dtype=dtype, opt=opt, rtol=self._RTOL[A_sparse.dtype])
        # Test with output matrix (C-contiguous) (fails on GPU)
        out = torch.empty(A_sparse.shape[0], B_sparse.shape[0], dtype=A_sparse.dtype)
        if not cpu:
            with pytest.raises(RuntimeError):
                _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=self._RTOL[A_sparse.dtype])
        else:
            _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=self._RTOL[A_sparse.dtype])
        # Test with output matrix (F-contiguous)
        # noinspection PyUnresolvedReferences
        out = torch.empty(B_sparse.shape[0], A_sparse.shape[0], dtype=A_sparse.dtype).T
        _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=self._RTOL[A_sparse.dtype])

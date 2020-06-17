import numpy as np
import pytest
import torch

from falkon.kernels import GaussianKernel, LinearKernel, PolynomialKernel
from falkon.tests.conftest import memory_checker
from falkon.tests.helpers import (
    gen_random, gen_sparse_matrix,
    naive_gaussian_kernel, naive_linear_kernel, naive_polynomial_kernel,
)
from falkon.utils import decide_cuda


def numpy_to_torch_type(dt):
    if dt == np.float32:
        return torch.float32
    elif dt == np.float64:
        return torch.float64
    else:
        raise TypeError("Invalid numpy type %s" % (dt,))


def choose_on_dtype(dtype):
    if dtype == np.float64:
        return 1e-12
    else:
        return 1e-5


def _run_fmm_test(k_class, k_exp, A, B, out, dtype, rtol, opt):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A.astype(dtype, copy=False))
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B.astype(dtype, copy=False))
    if out is not None and isinstance(out, np.ndarray):
        out = torch.from_numpy(out.astype(dtype, copy=False))

    with memory_checker(opt) as new_opt:
        actual = k_class(A, B, out=out, opt=new_opt)

    np.testing.assert_allclose(k_exp, actual, rtol=rtol)
    if out is not None:
        # Check output pointers
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


@pytest.fixture(scope="module")
def Ac():
    n, d = (4000, 10)
    return gen_random(n, d, 'float64', False, seed=92)


@pytest.fixture(scope="module")
def Af():
    n, d = (4000, 10)
    return np.asfortranarray(gen_random(n, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def Bc():
    m, d = (2000, 10)
    return gen_random(m, d, 'float64', False, seed=92)


@pytest.fixture(scope="module")
def Bf():
    m, d = (2000, 10)
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
    n, d, density = 500, 20000, 1e-4
    A = gen_sparse_matrix(n, d, np.float64, density=density, seed=14)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="class")
def s_B():
    m, d, density = 550, 20000, 1e-4
    B = gen_sparse_matrix(m, d, np.float64, density=density, seed=14)
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


@pytest.fixture(scope="class")
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
@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda({}), reason="No GPU found.")])
], ids=["cpu", "gpu"])
class TestDenseFmm:
    basic_options = {
        'debug': True,
        'compute_arch_speed': False,
        'no_single_kernel': False,  # Allow f32 kernels to be computed in f32
    }

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("A,B", [
        pytest.param('Ac', 'Bc', marks=pytest.mark.usefixtures('Ac', 'Bc')),
        pytest.param('Af', 'Bf', marks=pytest.mark.usefixtures('Af', 'Bf')),
        pytest.param('Ac', 'Bf', marks=pytest.mark.usefixtures('Ac', 'Bf')),
    ], indirect=True)
    def test(self, A, B, k_class, k_exp, dtype, cpu):
        max_mem = 2 * 2 ** 20

        opt = dict(self.basic_options)
        opt['use_cpu'] = cpu
        opt['max_cpu_mem'] = max_mem
        opt['max_gpu_mem'] = max_mem

        rtol = choose_on_dtype(dtype)
        _run_fmm_test(k_class, k_exp, A, B, out=None, dtype=dtype, opt=opt, rtol=rtol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_with_out(self, Ac: torch.Tensor, Bc: torch.Tensor, k_class, k_exp, dtype, cpu):
        out = np.empty((Ac.shape[0], Bc.shape[0]), dtype=Ac.dtype)
        max_mem = 2 * 2 ** 20
        opt = dict(self.basic_options)
        opt['use_cpu'] = cpu
        opt['max_cpu_mem'] = max_mem
        opt['max_gpu_mem'] = max_mem
        rtol = choose_on_dtype(dtype)
        _run_fmm_test(k_class, k_exp, Ac, Bc, out=out, dtype=dtype, opt=opt, rtol=rtol)

    @pytest.mark.parametrize("A,B", [
        pytest.param('Af', 'Bf', marks=pytest.mark.usefixtures('Af', 'Bf')),
        pytest.param('Ac', 'Bf', marks=pytest.mark.usefixtures('Ac', 'Bf')),
    ], indirect=True)
    def test_precise_kernel(self, A, B, k_class, k_exp, cpu):
        max_mem = 2 * 2 ** 20
        opt = dict(self.basic_options)
        opt['use_cpu'] = cpu
        opt['max_cpu_mem'] = max_mem
        opt['max_gpu_mem'] = max_mem
        opt['no_single_kernel'] = True
        expected_rtol = 1e-6
        out = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
        _run_fmm_test(k_class, k_exp, A, B, out=out, dtype=np.float32, opt=opt, rtol=expected_rtol)


@pytest.mark.parametrize("k_class,k_exp", [
    pytest.param('k1', 's_expected1', marks=pytest.mark.usefixtures('k1', 's_expected1')),
    pytest.param('k2', 's_expected2', marks=pytest.mark.usefixtures('k2', 's_expected2')),
    pytest.param('k3', 's_expected3', marks=pytest.mark.usefixtures('k3', 's_expected3')),
], indirect=True)
@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda({}), reason="No GPU found.")])
])
class TestSparseFmm:
    basic_opt = {
        'debug': True,
        'compute_arch_speed': False,
        'no_single_kernel': True,
    }

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sparse(self, k_class, k_exp, s_A, s_B, dtype, cpu):
        opt = dict(self.basic_opt)
        opt['max_gpu_mem'] = 50 * 2 ** 20
        opt['max_cpu_mem'] = 50 * 2 ** 20
        opt['use_cpu'] = cpu

        A_sparse = s_A[0].to(dtype=numpy_to_torch_type(dtype))
        B_sparse = s_B[0].to(dtype=numpy_to_torch_type(dtype))
        rtol = choose_on_dtype(dtype)

        # Here both A and B are sparse
        _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=None, dtype=dtype, opt=opt, rtol=rtol)
        # Test with output matrix (C) (fails on GPU)
        out = torch.empty(A_sparse.shape[0], B_sparse.shape[0], dtype=A_sparse.dtype)
        if not cpu:
            with pytest.raises(RuntimeError):
                _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=rtol)
        else:
            _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=rtol)
        # Test with output matrix (F)
        out = torch.empty(B_sparse.shape[0], A_sparse.shape[0], dtype=A_sparse.dtype).T
        _run_fmm_test(k_class, k_exp, A_sparse, B_sparse, out=out, dtype=dtype, opt=opt, rtol=rtol)

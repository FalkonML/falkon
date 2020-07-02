import dataclasses
from typing import Tuple

import numpy as np
import pytest
import torch
from falkon.options import FalkonOptions
from pytest import mark

from falkon.kernels import GaussianKernel, LinearKernel, PolynomialKernel
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.tests.conftest import memory_checker, fix_mat
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.utils import decide_cuda

n32 = np.float32
n64 = np.float64


def choose_on_dtype(dtype):
    if dtype == np.float64 or dtype == torch.float64:
        return 1e-12
    else:
        return 1e-5


def numpy_to_torch_type(dt):
    if dt == np.float32:
        return torch.float32
    elif dt == np.float64:
        return torch.float64
    else:
        raise TypeError("Invalid numpy type %s" % (dt,))


def _run_fmmv_test(fn, exp, tensors, out, rtol, opt):
    with memory_checker(opt) as new_opt:
        actual = fn(*tensors, out=out, opt=new_opt)

    # Check 1. Accuracy
    np.testing.assert_allclose(exp, actual.cpu(), rtol=rtol)
    # Check 2. Output pointers
    if out is not None:
        assert out.data_ptr() == actual.data_ptr(), "Output data tensor was not used"


@pytest.fixture(scope="module")
def n():
    return 4000

@pytest.fixture(scope="module")
def m():
    return 2000

@pytest.fixture(scope="module")
def d():
    return 10


@pytest.fixture(scope="module")
def t():
    return 5

@pytest.fixture(scope="module")
def A(n, d):
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def getA(A, s_A: Tuple[SparseTensor, torch.Tensor]):
    def convert(dtype, order=None, sparse=False, cuda=False):
        if sparse:
            out = s_A[0].to(dtype=numpy_to_torch_type(dtype))
        else:
            out = fix_mat(A, dtype=dtype, order=order)
        if out is not None and cuda:
            return out.cuda()
        return out

    return convert


@pytest.fixture(scope="module")
def B(m, d):
    return torch.from_numpy(gen_random(m, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def getB(B, s_B: Tuple[SparseTensor, torch.Tensor]):
    def convert(dtype, order=None, sparse=False, cuda=False):
        if sparse:
            out = s_B[0].to(dtype=numpy_to_torch_type(dtype))
        else:
            out = fix_mat(B, dtype=dtype, order=order)
        if out is not None and cuda:
            return out.cuda()
        return out

    return convert


@pytest.fixture(scope="module")
def v(m, t):
    return torch.from_numpy(gen_random(m, t, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def getv(v):
    def convert(dtype, order, cuda=False):
        out = fix_mat(v, dtype=dtype, order=order)
        if out is not None and cuda:
            return out.cuda()
        return out

    return convert


@pytest.fixture(scope="module")
def w(n, t):
    return torch.from_numpy(gen_random(n, t, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def getw(w):
    def convert(dtype, order, cuda=False):
        out = fix_mat(w, dtype=dtype, order=order)
        if out is not None and cuda:
            return out.cuda()
        return out

    return convert


@pytest.fixture(scope="module", params=[1, 2, 3], ids=["Gaussian", "Linear", "Polynomial"])
def kernel(request):
    if request.param == 1:
        return GaussianKernel(sigma=1)
    elif request.param == 2:
        return LinearKernel()
    elif request.param == 3:
        return PolynomialKernel(1.2, 3, 2.5)


@pytest.fixture(scope="module")
def gram(kernel, A, B):
    opt = FalkonOptions(use_cpu=True, compute_arch_speed=False)
    return kernel(A, B, opt=opt)


@pytest.fixture(scope="module")
def expected_fmmv(gram, v):
    return gram @ v


@pytest.fixture(scope="module")
def e_dfmmv1(gram, v, w):
    return gram.T @ (gram @ v + w)


@pytest.fixture(scope="module")
def e_dfmmv2(gram, v):
    return gram.T @ (gram @ v)


@pytest.fixture(scope="module")
def e_dfmmv3(gram, w):
    return gram.T @ w


@pytest.fixture(scope="module")
def e_dfmmv(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
], ids=["cpu", "gpu"])
class TestDense:
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="no")

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("F", np.float32, "F", np.float32, "F", np.float32),
        ("C", np.float32, "C", np.float32, "C", np.float32),
        ("F", np.float64, "F", np.float64, "F", np.float64),
        ("C", np.float64, "C", np.float64, "C", np.float64),
        # A few mixed-contiguity examples
        ("F", np.float32, "C", np.float32, "F", np.float32),
        ("F", np.float32, "C", np.float32, "C", np.float32),
    ], ids=["AF32-BF32-vF32", "AC32-BC32-vC32", "AF64-BF64-vF64", "AC64-BC64-vC64",
            "AF32-BC32-vF32", "AF32-BC32-vC32"])
    @pytest.mark.parametrize("max_mem", [2 * 2 ** 20])
    @pytest.mark.parametrize("cuda_inputs", [True, False], ids=["CUDA inputs", "CPU inputs"])
    def test_fmmv(self, getA, getB, getv, Ao, Adt, Bo, Bdt, vo, vdt, kernel,
                  expected_fmmv, max_mem, cpu, cuda_inputs):
        if cuda_inputs and cpu:
            return True
        A = getA(order=Ao, dtype=Adt, cuda=cuda_inputs)
        B = getB(order=Bo, dtype=Bdt, cuda=cuda_inputs)
        v = getv(order=vo, dtype=vdt, cuda=cuda_inputs)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, max_cpu_mem=max_mem, max_gpu_mem=max_mem)
        rtol = choose_on_dtype(A.dtype)

        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        if cuda_inputs:
            out = out.cuda()
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol, opt=opt)

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt,wo,wdt,e_dfmmv", [
        pytest.param("F", n32, "F", n32, "F", n32, "F", n32, "e_dfmmv1", marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("C", n32, "C", n32, "C", n32, "C", n32, "e_dfmmv1", marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("F", n64, "F", n64, "F", n64, "F", n64, "e_dfmmv1", marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("C", n64, "C", n64, "C", n64, "C", n64, "e_dfmmv1", marks=mark.usefixtures("e_dfmmv1")),
        pytest.param("F", n32, "F", n32, "F", n32, None, None, "e_dfmmv2", marks=mark.usefixtures("e_dfmmv2")),
        pytest.param("C", n32, "C", n32, "C", n32, None, None, "e_dfmmv2", marks=mark.usefixtures("e_dfmmv2")),
        pytest.param("F", n64, "F", n64, "F", n64, None, None, "e_dfmmv2", marks=mark.usefixtures("e_dfmmv2")),
        pytest.param("C", n64, "C", n64, "C", n64, None, None, "e_dfmmv2", marks=mark.usefixtures("e_dfmmv2")),
        pytest.param("F", n32, "F", n32, None, None, "F", n32, "e_dfmmv3", marks=mark.usefixtures("e_dfmmv3")),
        pytest.param("C", n32, "C", n32, None, None, "C", n32, "e_dfmmv3", marks=mark.usefixtures("e_dfmmv3")),
        pytest.param("F", n64, "F", n64, None, None, "F", n64, "e_dfmmv3", marks=mark.usefixtures("e_dfmmv3")),
        pytest.param("C", n64, "C", n64, None, None, "C", n64, "e_dfmmv3", marks=mark.usefixtures("e_dfmmv3")),
        # A few mixed-contiguity examples
        pytest.param("F", n32, "C", n32, "C", n32, "F", n32, "e_dfmmv1", marks=mark.usefixtures("e_dfmmv1")),
    ], ids=["F32-F32-vF32-wF32", "C32-C32-vC32-wC32", "F64-F64-vF64-wF64", "C64-C64-vC64-wC64",
            "F32-F32-vF32", "C32-C32-vC32", "F64-F64-vF64", "C64-C64-vC64",
            "F32-F32-wF32", "C32-C32-wC32", "F64-F64-wF64", "C64-C64-wC64",
            "F32-C32-vC32-wF32"],
       indirect=["e_dfmmv"])
    @pytest.mark.parametrize("max_mem", [2 * 2 ** 20])
    @pytest.mark.parametrize("cuda_inputs", [True, False], ids=["CUDA inputs", "CPU inputs"])
    def test_dfmmv(self, getA, getB, getv, getw, Ao, Adt, Bo, Bdt, vo, vdt, wo, wdt, kernel,
                   e_dfmmv, max_mem, cpu, m, t, cuda_inputs):
        if cuda_inputs and cpu:
            return True
        A = getA(order=Ao, dtype=Adt, cuda=cuda_inputs)
        B = getB(order=Bo, dtype=Bdt, cuda=cuda_inputs)
        v = getv(order=vo, dtype=vdt, cuda=cuda_inputs)
        w = getw(order=wo, dtype=wdt, cuda=cuda_inputs)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, max_cpu_mem=max_mem, max_gpu_mem=max_mem)
        rtol = choose_on_dtype(A.dtype)

        # Test normal
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype)
        if cuda_inputs:
            out = out.cuda()
        _run_fmmv_test(kernel.dmmv, e_dfmmv, (A, B, v, w), out=out, rtol=rtol, opt=opt)


@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
], ids=["cpu", "gpu"])
class TestKeops:
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="force")

    @pytest.mark.parametrize("Ao,Adt,Bo,Bdt,vo,vdt", [
        ("C", np.float32, "C", np.float32, "C", np.float32),
        ("C", np.float64, "C", np.float64, "C", np.float64),
        pytest.param("F", np.float32, "F", np.float32, "F", np.float32, marks=[pytest.mark.xfail(reason="KeOps only C")]),
        pytest.param("F", np.float32, "C", np.float32, "C", np.float32, marks=[pytest.mark.xfail(reason="KeOps only C")]),
    ], ids=["AC32-BC32-vC32", "AC64-BC64-vC64", "AF32-BF32-vF32", "AF32-BC32-vC32"])
    @pytest.mark.parametrize("max_mem", [2 * 2 ** 20])
    def test_fmmv(self, getA, getB, getv, Ao, Adt, Bo, Bdt, vo, vdt, kernel,
                  expected_fmmv, max_mem, cpu):
        A = getA(order=Ao, dtype=Adt)
        B = getB(order=Bo, dtype=Bdt)
        v = getv(order=vo, dtype=vdt)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, max_cpu_mem=max_mem, max_gpu_mem=max_mem)
        rtol = choose_on_dtype(A.dtype)

        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol, opt=opt)

    def test_gpu_inputs(self, getA, getB, getv, kernel, expected_fmmv, cpu):
        if cpu:  # No point in testing GPU inputs without GPU
            return True
        A = getA(order="C", dtype=n32).cuda()
        B = getB(order="C", dtype=n32).cuda()
        v = getv(order="C", dtype=n32).cuda()
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        rtol = choose_on_dtype(A.dtype)
        # Test normal
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype).cuda()
        _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=out, rtol=rtol, opt=opt)

    def test_gpu_inputs_fail(self, getA, getB, getv, kernel, expected_fmmv, cpu):
        if cpu:  # No point in testing GPU inputs without GPU
            return True
        A = getA(order="C", dtype=n32).cuda()
        B = getB(order="C", dtype=n32).cuda()
        v = getv(order="C", dtype=n32)
        opt = dataclasses.replace(self.basic_options, use_cpu=cpu)
        rtol = choose_on_dtype(A.dtype)
        # Test normal
        with pytest.raises(RuntimeError):
            _run_fmmv_test(kernel.mmv, expected_fmmv, (A, B, v), out=None, rtol=rtol, opt=opt)


###################
### Sparse Test ###
###################
@pytest.fixture(scope="module")
def s_d():
    return 10_000


@pytest.fixture(scope="module")
def s_density():
    return 1e-4


@pytest.fixture(scope="module")
def s_A(n, s_d, s_density):
    A = gen_sparse_matrix(n, s_d, np.float64, density=s_density, seed=14)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="module")
def s_B(m, s_d, s_density):
    B = gen_sparse_matrix(m, s_d, np.float64, density=s_density, seed=14)
    Bd = torch.from_numpy(B.to_scipy().todense())
    return B, Bd


@pytest.fixture(scope="module")
def s_gram(kernel, s_A, s_B):
    opt = FalkonOptions(use_cpu=True, compute_arch_speed=False)
    return kernel(s_A[1], s_B[1], opt=opt)


@pytest.fixture(scope="module")
def s_expected_fmmv(s_gram, v):
    return s_gram @ v


@pytest.fixture(scope="module")
def s_e_dfmmv1(s_gram, v, w):
    return s_gram.T @ (s_gram @ v + w)


@pytest.fixture(scope="module")
def s_e_dfmmv2(s_gram, v):
    return s_gram.T @ (s_gram @ v)


@pytest.fixture(scope="module")
def s_e_dfmmv3(s_gram, w):
    return s_gram.T @ w


@pytest.fixture(scope="module")
def s_e_dfmmv(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("cpu", [
    pytest.param(True),
    pytest.param(False, marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
], ids=["cpu", "gpu"])
class TestSparse:
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False)

    @pytest.mark.parametrize("Adt,Bdt,vo,vdt", [
        (np.float32, np.float32, "F", np.float32),
        (np.float32, np.float32, "C", np.float32),
        (np.float64, np.float64, "F", np.float64),
        (np.float64, np.float64, "C", np.float64),
    ], ids=["A32-B32-vF32", "A32-B32-vC32", "A64-B64-vF64", "A64-B64-vC64"])
    @pytest.mark.parametrize("max_mem", [2 * 2 ** 20])
    def test_fmmv(self, getA, getB, getv, Adt, Bdt, vo, vdt, kernel,
                  s_expected_fmmv, max_mem, cpu):
        A = getA(dtype=Adt, sparse=True)
        B = getB(dtype=Bdt, sparse=True)
        v = getv(order=vo, dtype=vdt)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, max_cpu_mem=max_mem, max_gpu_mem=max_mem)
        rtol = choose_on_dtype(A.dtype)

        # Test normal
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(A.shape[0], v.shape[1], dtype=A.dtype)
        _run_fmmv_test(kernel.mmv, s_expected_fmmv, (A, B, v), out=out, rtol=rtol, opt=opt)

    @pytest.mark.parametrize("Adt,Bdt,vo,vdt,wo,wdt,s_e_dfmmv", [
        pytest.param(n32, n32, "F", n32, "F", n32, "s_e_dfmmv1", marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n32, n32, "C", n32, "C", n32, "s_e_dfmmv1", marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n64, n64, "F", n64, "F", n64, "s_e_dfmmv1", marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n64, n64, "C", n64, "C", n64, "s_e_dfmmv1", marks=mark.usefixtures("s_e_dfmmv1")),
        pytest.param(n32, n32, "F", n32, None, None, "s_e_dfmmv2", marks=mark.usefixtures("s_e_dfmmv2")),
        pytest.param(n32, n32, "C", n32, None, None, "s_e_dfmmv2", marks=mark.usefixtures("s_e_dfmmv2")),
        pytest.param(n64, n64, "F", n64, None, None, "s_e_dfmmv2", marks=mark.usefixtures("s_e_dfmmv2")),
        pytest.param(n64, n64, "C", n64, None, None, "s_e_dfmmv2", marks=mark.usefixtures("s_e_dfmmv2")),
        pytest.param(n32, n32, None, None, "F", n32, "s_e_dfmmv3", marks=mark.usefixtures("s_e_dfmmv3")),
        pytest.param(n32, n32, None, None, "C", n32, "s_e_dfmmv3", marks=mark.usefixtures("s_e_dfmmv3")),
        pytest.param(n64, n64, None, None, "F", n64, "s_e_dfmmv3", marks=mark.usefixtures("s_e_dfmmv3")),
        pytest.param(n64, n64, None, None, "C", n64, "s_e_dfmmv3", marks=mark.usefixtures("s_e_dfmmv3")),
        # A few mixed-contiguity examples
        pytest.param(n32, n32, "C", n32, "F", n32, "s_e_dfmmv1", marks=mark.usefixtures("s_e_dfmmv1")),
    ], ids=["32-32-vF32-wF32", "32-32-vC32-wC32", "64-64-vF64-wF64", "64-64-vC64-wC64",
            "32-32-vF32", "32-32-vC32", "64-64-vF64", "64-64-vC64",
            "32-32-wF32", "32-32-wC32", "64-64-wF64", "64-64-wC64",
            "32-32-vC32-wF32"
            ],
       indirect=["s_e_dfmmv"])
    @pytest.mark.parametrize("max_mem", [2 * 2 ** 20])
    def test_dfmmv(self, getA, getB, getv, getw, Adt, Bdt, vo, vdt, wo, wdt, kernel,
                   s_e_dfmmv, max_mem, cpu, m, t):
        A = getA(dtype=Adt, sparse=True)
        B = getB(dtype=Bdt, sparse=True)
        v = getv(order=vo, dtype=vdt)
        w = getw(order=wo, dtype=wdt)

        opt = dataclasses.replace(self.basic_options, use_cpu=cpu, max_cpu_mem=max_mem, max_gpu_mem=max_mem)
        rtol = choose_on_dtype(A.dtype)

        # Test normal
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=None, rtol=rtol, opt=opt)
        # Test with out
        out = torch.empty(m, t, dtype=A.dtype)
        _run_fmmv_test(kernel.dmmv, s_e_dfmmv, (A, B, v, w), out=out, rtol=rtol, opt=opt)

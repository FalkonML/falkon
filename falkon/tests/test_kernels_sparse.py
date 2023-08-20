import dataclasses

import numpy as np
import pytest
import torch

from falkon.kernels import GaussianKernel, LaplacianKernel, MaternKernel, PolynomialKernel
from falkon.mmv_ops.utils import CUDA_EXTRA_MM_RAM
from falkon.options import FalkonOptions
from falkon.tests.conftest import fix_mats, memory_checker
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.tests.naive_kernels import (
    naive_diff_gaussian_kernel,
    naive_diff_laplacian_kernel,
    naive_diff_matern_kernel,
    naive_diff_polynomial_kernel,
)
from falkon.utils import decide_cuda
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.switches import decide_keops

cuda_mark = pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
keops_mark = pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")
device_marks = [
    pytest.param("cpu", "cpu"),
    pytest.param("cpu", "cuda", marks=[cuda_mark]),
    pytest.param(
        "cuda",
        "cuda",
        marks=[
            cuda_mark,
            pytest.mark.xfail(
                raises=NotImplementedError,
                strict=True,
                reason="Sparse kernels are not implemented for in-core CUDA operations",
            ),
        ],
    ),
]
# Sparse data dimensions
n = 500
m = 550
d = 20000
t = 2
density = 1e-5

max_mem = 2 * 2**20
basic_options = FalkonOptions(debug=True, compute_arch_speed=False, max_cpu_mem=max_mem, max_gpu_mem=max_mem)


@pytest.fixture(scope="module")
def s_A():
    A = gen_sparse_matrix(n, d, np.float64, density=density, seed=14)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="module")
def s_B():
    B = gen_sparse_matrix(m, d, np.float64, density=density, seed=14)
    Bd = torch.from_numpy(B.to_scipy().todense())
    return B, Bd


@pytest.fixture(scope="module")
def v() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, t, "float32", False, seed=94))


@pytest.fixture(scope="module")
def w() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, t, "float32", False, seed=95))


@pytest.fixture(
    params=[
        "single-sigma",
        pytest.param(
            "vec-sigma",
            marks=pytest.mark.xfail(
                raises=NotImplementedError,
                strict=True,
                reason="Sparse kernels are not implemented for vectorial sigmas",
            ),
        ),
    ],
    scope="class",
)
def sigma(request) -> torch.Tensor:
    if request.param == "single-sigma":
        return torch.Tensor([3.0])
    elif request.param == "vec-sigma":
        return torch.Tensor([3.0] * d)


@pytest.fixture(scope="module")
def rtol():
    return {np.float64: 1e-12, torch.float64: 1e-12, np.float32: 1e-4, torch.float32: 1e-4}


@pytest.fixture(scope="module")
def atol():
    return {np.float64: 1e-12, torch.float64: 1e-12, np.float32: 1e-4, torch.float32: 1e-4}


def fix_options(opt):
    return dataclasses.replace(
        opt,
        max_cpu_mem=opt.max_cpu_mem,
        max_gpu_mem=opt.max_gpu_mem + CUDA_EXTRA_MM_RAM,
    )


def run_sparse_test(k_cls, naive_fn, s_m1, s_m2, m1, m2, v, w, rtol, atol, opt, **kernel_params):
    kernel = k_cls(**kernel_params)
    opt = fix_options(opt)
    print(f"max mem: {opt.max_gpu_mem}")

    # 1. MM
    mm_out = torch.empty(s_m2.shape[0], s_m1.shape[0], dtype=s_m1.dtype, device=s_m1.device).T
    with memory_checker(opt) as new_opt:
        actual = kernel(s_m1, s_m2, out=mm_out, opt=new_opt)
    with memory_checker(opt, extra_mem=m1.shape[0] * m2.shape[0] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel(s_m1, s_m2, opt=new_opt)
    assert mm_out.data_ptr() == actual.data_ptr(), "sparse MM Output data tensor was not used"
    torch.testing.assert_close(
        actual_noout, actual, rtol=rtol, atol=atol, msg="sparse MM with out and without return different stuff"
    )
    expected_mm = naive_fn(m1, m2, **kernel_params)
    torch.testing.assert_close(expected_mm, actual, rtol=rtol, atol=atol, msg="sparse MM result is incorrect")

    # 2. MMV
    mmv_out = torch.empty(s_m1.shape[0], v.shape[1], dtype=s_m1.dtype, device=s_m1.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.mmv(s_m1, s_m2, v, out=mmv_out, opt=new_opt)
    with memory_checker(opt, extra_mem=m1.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.mmv(s_m1, s_m2, v, opt=new_opt)
    assert mmv_out.data_ptr() == actual.data_ptr(), "sparse MMV Output data tensor was not used"
    torch.testing.assert_close(
        actual_noout, actual, rtol=rtol, atol=atol, msg="sparse MMV with out and without return different stuff"
    )
    expected_mmv = expected_mm @ v
    torch.testing.assert_close(expected_mmv, actual, rtol=rtol, atol=atol, msg="sparse MMV result is incorrect")

    # 3. dMMV
    dmmv_out = torch.empty(s_m2.shape[0], v.shape[1], dtype=s_m2.dtype, device=s_m2.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.dmmv(s_m1, s_m2, v, w, out=dmmv_out, opt=new_opt)
    with memory_checker(opt, extra_mem=m2.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.dmmv(s_m1, s_m2, v, w, opt=new_opt)
    assert dmmv_out.data_ptr() == actual.data_ptr(), "sparse D-MMV Output data tensor was not used"
    torch.testing.assert_close(
        actual_noout, actual, rtol=rtol, atol=atol, msg="sparse D-MMV with out and without return different stuff"
    )
    expected_dmmv = expected_mm.T @ (expected_mmv + w)
    torch.testing.assert_close(expected_dmmv, actual, rtol=rtol, atol=atol, msg="sparse D-MMV result is incorrect")


def run_sparse_test_wsigma(k_cls, naive_fn, s_m1, s_m2, m1, m2, v, w, rtol, atol, opt, sigma, **kernel_params):
    try:
        run_sparse_test(
            k_cls,
            naive_fn,
            s_m1=s_m1,
            s_m2=s_m2,
            m1=m1,
            m2=m2,
            v=v,
            w=w,
            rtol=rtol,
            atol=atol,
            opt=opt,
            sigma=sigma,
            **kernel_params,
        )
    except Exception as e:
        # Always raise the base exception
        if hasattr(e, "__cause__") and e.__cause__ is not None:
            raise e.__cause__ from e
        raise e


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestLaplacianKernel:
    naive_fn = naive_diff_laplacian_kernel
    k_class = LaplacianKernel

    def test_sparse_kernel(self, s_A, s_B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, v, w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, v, w, sigma, order="C", device=input_dev, dtype=np.float32
        )
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_sparse_test_wsigma(
            TestLaplacianKernel.k_class,
            TestLaplacianKernel.naive_fn,
            s_m1=s_A,
            s_m2=s_B,
            m1=A,
            m2=B,
            v=v,
            w=w,
            rtol=rtol[A.dtype],
            atol=atol[A.dtype],
            opt=opt,
            sigma=sigma,
        )


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestGaussianKernel:
    naive_fn = naive_diff_gaussian_kernel
    k_class = GaussianKernel

    def test_sparse_kernel(self, s_A, s_B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, v, w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, v, w, sigma, order="C", device=input_dev, dtype=np.float32
        )
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_sparse_test_wsigma(
            TestGaussianKernel.k_class,
            TestGaussianKernel.naive_fn,
            s_m1=s_A,
            s_m2=s_B,
            m1=A,
            m2=B,
            v=v,
            w=w,
            rtol=rtol[A.dtype],
            atol=atol[A.dtype],
            opt=opt,
            sigma=sigma,
        )


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestMaternKernel:
    naive_fn = naive_diff_matern_kernel
    k_class = MaternKernel

    @pytest.fixture(params=[0.5, 1.5, 2.5, np.inf], scope="function")
    def nu(self, request) -> torch.Tensor:
        return torch.tensor(request.param)

    def test_sparse_kernel(self, s_A, s_B, v, w, sigma, nu, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, v, w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, v, w, sigma, order="C", device=input_dev, dtype=np.float32
        )
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_sparse_test_wsigma(
            TestMaternKernel.k_class,
            TestMaternKernel.naive_fn,
            s_m1=s_A,
            s_m2=s_B,
            m1=A,
            m2=B,
            v=v,
            w=w,
            rtol=rtol[A.dtype],
            atol=atol[A.dtype],
            opt=opt,
            sigma=sigma,
            nu=nu,
        )


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestPolynomialKernel:
    naive_fn = naive_diff_polynomial_kernel
    k_class = PolynomialKernel
    beta = torch.tensor(1.0)
    gamma = torch.tensor(2.0)
    degree = torch.tensor(1.5)

    def test_sparse_kernel(self, s_A, s_B, v, w, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, v, w, beta, gamma, degree = fix_mats(
            s_A, d_A, s_B, d_B, v, w, self.beta, self.gamma, self.degree, order="C", device=input_dev, dtype=np.float32
        )
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        try:
            run_sparse_test(
                TestPolynomialKernel.k_class,
                TestPolynomialKernel.naive_fn,
                s_m1=s_A,
                s_m2=s_B,
                m1=A,
                m2=B,
                v=v,
                w=w,
                rtol=rtol[A.dtype],
                atol=atol[A.dtype],
                opt=opt,
                beta=beta,
                gamma=gamma,
                degree=degree,
            )
        except Exception as e:
            # Always raise the base exception
            if hasattr(e, "__cause__") and e.__cause__ is not None:
                raise e.__cause__ from None
            raise e

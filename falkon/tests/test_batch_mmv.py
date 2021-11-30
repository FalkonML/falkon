import dataclasses
import time

import pytest

import numpy as np
import torch

from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from falkon.utils import decide_cuda
from falkon.mmv_ops.utils import _gpu_tns_same_memory
from falkon.mmv_ops.batch_mmv import batch_fmmv_incore, batch_fmmv_ooc
from falkon.utils.helpers import sizeof_dtype
from falkon.tests.conftest import fix_mat, memory_checker
from falkon.tests.gen_random import gen_random_multi


@pytest.fixture(scope="module")
def rtol() -> dict:
    return {
        torch.float32: 1e-4,
        np.float32: 1e-4,
        torch.float64: 1e-12,
        np.float64: 1e-12,
    }


def gen_data(b, n, d, m, t):
    return (torch.from_numpy(gen_random_multi(b, n, d, dtype='float64', F=False, seed=92)),
            torch.from_numpy(gen_random_multi(b, m, d, dtype='float64', F=False, seed=92)),
            torch.from_numpy(gen_random_multi(b, m, t, dtype='float64', F=False, seed=92)),
            torch.from_numpy(gen_random_multi(b, n, t, dtype='float64', F=False, seed=92)))


def naive_batch_kernel(kernel, A, B, opt):
    num_batches = A.shape[0]
    k_out = []
    for i in range(num_batches):
        k_out.append(kernel(A[i], B[i], opt=opt))
    return torch.stack(k_out, dim=0)


@pytest.fixture(scope="module")
def kernel():
    return GaussianKernel(sigma=1)


class TestDimSelect():
    pass


@pytest.mark.parametrize("orderA,orderB,orderV,orderO", [
    pytest.param("F", "F", "F", "F"),
    pytest.param("C", "C", "C", "C", marks=pytest.mark.full()),
    pytest.param("C", "C", "F", "C", marks=pytest.mark.full()),
    pytest.param("F", "C", "F", "C", marks=pytest.mark.full()),
])
@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
@pytest.mark.parametrize("out", [True, False], ids=["out", "no-out"])
class TestBatchMmv:
    max_mem = 0.1 * 2**20
    data = gen_data(b=20, n=100, d=10, m=400, t=5)
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False, keops_active="no",
                                  max_gpu_mem=max_mem, max_cpu_mem=max_mem)

    def fix_mats(self, oa, ob, ov, oo, dt, dev, out=False):
        A = fix_mat(TestBatchMmv.data[0], dtype=dt, order=oa, device=dev)
        B = fix_mat(TestBatchMmv.data[1], dtype=dt, order=ob, device=dev)
        v = fix_mat(TestBatchMmv.data[2], dtype=dt, order=ov, device=dev)
        o = None
        if out:
            o = fix_mat(TestBatchMmv.data[3], dtype=dt, order=oo, device=dev)
        return A, B, v, o

    @pytest.fixture(scope="class")
    def expected(self, kernel):
        A, B, v, o = self.fix_mats("C", "C", "C", "C", np.float64, "cpu")
        opt = FalkonOptions(use_cpu=True, compute_arch_speed=False)
        K = naive_batch_kernel(kernel, A, B, opt)
        return torch.bmm(K, v)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_incore(self, orderA, orderB, orderV, orderO, dtype, out, kernel, expected, rtol):
        A, B, v, o = self.fix_mats(orderA, orderB, orderV, orderO, dtype, "cuda:0", out)
        extra_mem = 0
        if not out:
            extra_mem += A.shape[0] * A.shape[1] * v.shape[2] * sizeof_dtype(dtype)

        opt = dataclasses.replace(self.basic_options)
        with memory_checker(opt, extra_mem=extra_mem) as new_opt:
            out = batch_fmmv_incore(A, B, v, kernel, o, new_opt)
        torch.testing.assert_allclose(expected.to(dtype=out.dtype), out.cpu(), rtol=rtol[dtype], atol=0)
        if o is not None:
            assert _gpu_tns_same_memory(out, o), "output tensor was not used correctly"

    def test_cpu_incore(self, orderA, orderB, orderV, orderO, dtype, out, kernel, expected, rtol):
        A, B, v, o = self.fix_mats(orderA, orderB, orderV, orderO, dtype, "cpu", out)

        opt = dataclasses.replace(self.basic_options, use_cpu=True)
        with memory_checker(opt) as new_opt:
            out = batch_fmmv_incore(A, B, v, kernel, o, new_opt)
        torch.testing.assert_allclose(expected.to(dtype=out.dtype), out.cpu(), rtol=rtol[dtype], atol=0)
        if o is not None:
            assert _gpu_tns_same_memory(out, o), "output tensor was not used correctly"

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_ooc(self, orderA, orderB, orderV, orderO, dtype, out, kernel, expected, rtol):
        A, B, v, o = self.fix_mats(orderA, orderB, orderV, orderO, dtype, "cpu", out)

        opt = dataclasses.replace(self.basic_options)
        with memory_checker(opt) as new_opt:
            out = batch_fmmv_ooc(A, B, v, kernel, o, new_opt)
        torch.testing.assert_allclose(expected.to(dtype=out.dtype), out.cpu(), rtol=rtol[dtype], atol=0)
        if o is not None:
            assert _gpu_tns_same_memory(out, o), "output tensor was not used correctly"

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cpu_ooc(self, orderA, orderB, orderV, orderO, dtype, out, kernel, expected, rtol):
        A, B, v, o = self.fix_mats(orderA, orderB, orderV, orderO, dtype, "cuda", out)

        with pytest.raises(RuntimeError):
            batch_fmmv_ooc(A, B, v, kernel, o, self.basic_options)


@pytest.mark.full
def test_different_dtypes(kernel):
    data = gen_data(b=20, n=100, d=10, m=400, t=5)
    A = fix_mat(data[0], dtype=np.float32, order="F", device="cpu")
    B = fix_mat(data[1], dtype=np.float64, order="F", device="cpu")
    v = fix_mat(data[2], dtype=np.float32, order="F", device="cpu")
    opt = FalkonOptions()
    with pytest.raises(RuntimeError) as err_info:
        batch_fmmv_incore(A, B, v, kernel, None, opt)
    assert str(err_info.value).endswith(
        "expected scalar type Float but found Double")


@pytest.mark.full
def test_single_bnm(kernel, rtol):
    data = gen_data(b=1, n=1, d=1, m=1, t=1)
    A = fix_mat(data[0], dtype=np.float32, order="F", device="cpu")
    B = fix_mat(data[1], dtype=np.float32, order="F", device="cpu")
    v = fix_mat(data[2], dtype=np.float32, order="F", device="cpu")
    opt = FalkonOptions()

    K = naive_batch_kernel(kernel, A, B, opt)
    exp = torch.bmm(K, v)
    out = batch_fmmv_incore(A, B, v, kernel, None, opt)

    torch.testing.assert_allclose(exp.to(dtype=out.dtype), out.cpu(), rtol=rtol[A.dtype], atol=0)


@pytest.mark.benchmark
class TestBenchmark:
    num_rep = 25
    max_mem = 5 * 2**30
    data = gen_data(b=15, n=20000, d=2000, m=2000, t=15)
    basic_options = FalkonOptions(max_gpu_mem=max_mem, max_cpu_mem=max_mem)

    def fix_mats(self, oa, ob, ov, oo, dt, dev, out=False):
        A = fix_mat(TestBatchMmv.data[0], dtype=dt, order=oa, device=dev)
        B = fix_mat(TestBatchMmv.data[1], dtype=dt, order=ob, device=dev)
        v = fix_mat(TestBatchMmv.data[2], dtype=dt, order=ov, device=dev)
        o = None
        if out:
            o = fix_mat(TestBatchMmv.data[3], dtype=dt, order=oo, device=dev)
        return A, B, v, o

    def test_f_contig(self, kernel):
        A, B, v, o = self.fix_mats("F", "F", "F", "F", np.float32, "cpu", True)

        times = []
        for i in range(self.num_rep):
            t_s = time.time()
            batch_fmmv_incore(A, B, v, kernel, o, self.basic_options)
            times.append(time.time() - t_s)

        print("Timings: %.4f +- %.4f s" % (np.mean(times), np.std(times)))

    def test_C_contig(self, kernel):
        A, B, v, o = self.fix_mats("C", "C", "C", "C", np.float32, "cpu", True)

        times = []
        for i in range(self.num_rep):
            t_s = time.time()
            batch_fmmv_incore(A, B, v, kernel, o, self.basic_options)
            times.append(time.time() - t_s)

        print("Timings: %.4f +- %.4f s" % (np.mean(times), np.std(times)))

    def test_compare_cudaic_serial(self, kernel, rtol):
        data = gen_data(b=10, n=100, d=100, m=100, t=3)
        A = fix_mat(data[0], dtype=np.float32, order="F", device="cuda")
        B = fix_mat(data[1], dtype=np.float32, order="F", device="cuda")
        v = fix_mat(data[2], dtype=np.float32, order="F", device="cuda")
        o = fix_mat(data[3], dtype=np.float32, order="F", device="cuda")
        opt = FalkonOptions()

        for i in range(A.shape[0]):
            o[i] = kernel.mmv(A[i], B[i], v[i], opt=opt)

        out = batch_fmmv_incore(A, B, v, kernel, opt=opt)

        torch.testing.assert_allclose(o, out, rtol=rtol[A.dtype], atol=0)

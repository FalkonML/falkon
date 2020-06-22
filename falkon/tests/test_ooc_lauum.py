import dataclasses

import numpy as np
import pytest
import scipy.linalg.lapack as scll
import torch
from falkon.options import FalkonOptions

from falkon.tests.conftest import memory_checker, fix_mat
from falkon.utils import decide_cuda

if decide_cuda():
    from falkon.ooc_ops.ooc_lauum import gpu_lauum


N = 4000


@pytest.fixture(scope="module")
def matrix():
    # Output matrix in F-order
    return np.random.random((N, N)).T


@pytest.fixture(scope="module")
def get_mat(matrix):
    def getter(order, dtype):
        return fix_mat(matrix, dtype=dtype, order=order)
    return getter


@pytest.fixture(scope="module")
def expected_upper(matrix):
    return scll.dlauum(matrix, lower=0, overwrite_c=False)[0]


@pytest.fixture(scope="module")
def expected_lower(matrix):
    return scll.dlauum(matrix, lower=1, overwrite_c=False)[0]


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestOOCLauum:
    rtol = {
        np.float64: 1e-12,
        np.float32: 1e-5
    }
    basic_opt = FalkonOptions(compute_arch_speed=False, use_cpu=False, max_gpu_mem=2*2**20,
                              lauum_par_blk_multiplier=6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_no_overwrite(self, dtype, order, get_mat, expected_lower, expected_upper):
        mat = get_mat(order=order, dtype=dtype)

        with memory_checker(self.basic_opt) as new_opt:
            act_up = gpu_lauum(mat, upper=True, overwrite=False, opt=new_opt)
        np.testing.assert_allclose(expected_upper, act_up.numpy(), rtol=self.rtol[dtype])

        with memory_checker(self.basic_opt) as new_opt:
            act_lo = gpu_lauum(mat, upper=False, overwrite=False, opt=new_opt)
        np.testing.assert_allclose(expected_lower, act_lo.numpy(), rtol=self.rtol[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_overwrite(self, dtype, order, get_mat, expected_lower, expected_upper):
        mat = get_mat(order=order, dtype=dtype).numpy().copy(order="K")
        with memory_checker(self.basic_opt) as new_opt:
            act_up = gpu_lauum(torch.from_numpy(mat), upper=True, overwrite=True, opt=new_opt)
        np.testing.assert_allclose(expected_upper, act_up.numpy(), rtol=self.rtol[dtype])

        mat = get_mat(order=order, dtype=dtype).numpy().copy(order="K")
        with memory_checker(self.basic_opt) as new_opt:
            act_lo = gpu_lauum(torch.from_numpy(mat), upper=False, overwrite=True, opt=new_opt)
        np.testing.assert_allclose(expected_lower, act_lo.numpy(), rtol=self.rtol[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_write_opposite(self, dtype, order, get_mat, expected_lower, expected_upper):
        omat = get_mat(order=order, dtype=dtype).numpy()
        mat = torch.from_numpy(omat.copy(order="K"))
        with memory_checker(self.basic_opt) as new_opt:
            act_up = gpu_lauum(mat, upper=True, overwrite=True, write_opposite=True, opt=new_opt)
        np.testing.assert_allclose(np.triu(omat, k=1), np.triu(act_up.numpy(), k=1), rtol=self.rtol[dtype])
        np.testing.assert_allclose(np.tril(act_up.numpy()), np.triu(expected_upper).T, rtol=self.rtol[dtype])

        mat = torch.from_numpy(omat.copy(order="K"))
        with memory_checker(self.basic_opt) as new_opt:
            act_lo = gpu_lauum(mat, upper=False, overwrite=True, write_opposite=True, opt=new_opt)
        np.testing.assert_allclose(np.tril(omat, k=-1), np.tril(act_lo.numpy(), k=-1), rtol=self.rtol[dtype])
        np.testing.assert_allclose(np.triu(act_lo.numpy()), np.tril(expected_lower).T, rtol=self.rtol[dtype])

    def test_no_blk_mul(self, get_mat, expected_lower):
        dtype = np.float32
        mat = get_mat(order="F", dtype=dtype).numpy().copy(order="K")
        opt = dataclasses.replace(self.basic_opt, lauum_par_blk_multiplier=1)

        act_lo = gpu_lauum(torch.from_numpy(mat), upper=False, overwrite=True, opt=opt)
        np.testing.assert_allclose(expected_lower, act_lo.numpy(), rtol=self.rtol[dtype])

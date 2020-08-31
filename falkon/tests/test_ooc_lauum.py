import dataclasses

import numpy as np
import pytest
import scipy.linalg.lapack as scll
import torch

from falkon.options import FalkonOptions
from falkon.tests.conftest import memory_checker, fix_mat
from falkon.utils import decide_cuda
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.tensor_helpers import move_tensor

if decide_cuda():
    from falkon.ooc_ops.ooc_utils import calc_block_sizes3
    from falkon.ooc_ops.ooc_lauum import gpu_lauum
    # noinspection PyUnresolvedReferences
    from falkon.ooc_ops.cuda import cuda_lauum_lower


class TestBlockSizeCalculator:
    def test_small_edge(self):
        assert calc_block_sizes3(
            max_block_size=1, num_devices=4, num_rows=3) == [1, 1, 1]
        assert calc_block_sizes3(
            max_block_size=1, num_devices=5, num_rows=1) == [1]

    def test_small(self):
        assert calc_block_sizes3(
            max_block_size=10000, num_devices=2, num_rows=100) == [100]
        assert calc_block_sizes3(
            max_block_size=5, num_devices=2, num_rows=10) == [5, 5]
        assert calc_block_sizes3(
            max_block_size=6, num_devices=3, num_rows=10) == [4, 3, 3]

    def test_edge_preferred(self):
        assert calc_block_sizes3(
            max_block_size=10000, num_devices=2, num_rows=3068) == [1534, 1534]
        assert calc_block_sizes3(
            max_block_size=10000, num_devices=1, num_rows=7000) == [7000]
        assert calc_block_sizes3(
            max_block_size=10000, num_devices=1, num_rows=7001) == [3501, 3500]

    def test_max_block_size(self):
        assert calc_block_sizes3(
            max_block_size=50, num_devices=1, num_rows=101) == [34, 34, 33]
        assert calc_block_sizes3(
            max_block_size=50, num_devices=2, num_rows=101) == [26, 25, 25, 25]
        assert calc_block_sizes3(
            max_block_size=10000, num_devices=1, num_rows=10000) == [5000, 5000]

    def test_large(self):
        assert calc_block_sizes3(
            max_block_size=50000, num_devices=1, num_rows=50000) == [6250, 6250, 6250, 6250, 6250,
                                                                     6250, 6250, 6250]
        assert calc_block_sizes3(
            max_block_size=50000, num_devices=6, num_rows=50000) == [4167, 4167, 4167, 4167, 4167,
                                                                     4167, 4167, 4167, 4166, 4166,
                                                                     4166, 4166]


# Size of test matrix
N = 1500


@pytest.fixture(scope="module")
def matrix():
    # Output matrix in F-order
    np.random.seed(233)
    return np.random.random((N, N)).T


@pytest.fixture(scope="module")
def get_mat(matrix):
    def getter(order, dtype):
        return fix_mat(matrix, dtype=dtype, order=order, copy=True)

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
    max_mem = 2 * 2**20
    basic_opt = FalkonOptions(compute_arch_speed=False, use_cpu=False, max_gpu_mem=max_mem,
                              lauum_par_blk_multiplier=6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_no_overwrite(self, dtype, order, get_mat, expected_lower, expected_upper, device):
        mat = get_mat(order=order, dtype=dtype)
        mat = move_tensor(mat, device)

        # For cuda inputs we must add to available GPU memory the amount used by the
        # input matrix, since overwrite=False and a full copy must be performed.
        mgpu_slack = 0
        if device.startswith("cuda"):
            mgpu_slack = self.basic_opt.max_gpu_mem + mat.shape[0]**2 * sizeof_dtype(mat.dtype)

        with memory_checker(self.basic_opt, extra_mem=mgpu_slack) as new_opt:
            act_up = gpu_lauum(mat, upper=True, overwrite=False, opt=new_opt)
            torch.cuda.synchronize()
        np.testing.assert_allclose(expected_upper, act_up.cpu().numpy(), rtol=self.rtol[dtype])

        with memory_checker(self.basic_opt, extra_mem=mgpu_slack) as new_opt:
            act_lo = gpu_lauum(mat, upper=False, overwrite=False, opt=new_opt)
            torch.cuda.synchronize()
        np.testing.assert_allclose(expected_lower, act_lo.cpu().numpy(), rtol=self.rtol[dtype])

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
            torch.cuda.synchronize()
        np.testing.assert_allclose(np.triu(omat, k=1), np.triu(act_up.numpy(), k=1),
                                   rtol=self.rtol[dtype])
        np.testing.assert_allclose(np.tril(act_up.numpy()), np.triu(expected_upper).T,
                                   rtol=self.rtol[dtype])

        mat = torch.from_numpy(omat.copy(order="K"))
        with memory_checker(self.basic_opt) as new_opt:
            act_lo = gpu_lauum(mat, upper=False, overwrite=True, write_opposite=True, opt=new_opt)
            torch.cuda.synchronize()
        np.testing.assert_allclose(np.tril(omat, k=-1), np.tril(act_lo.numpy(), k=-1),
                                   rtol=self.rtol[dtype])
        np.testing.assert_allclose(np.triu(act_lo.numpy()), np.tril(expected_lower).T,
                                   rtol=self.rtol[dtype])

    def test_no_blk_mul(self, get_mat, expected_upper):
        dtype = np.float32
        mat = get_mat(order="F", dtype=dtype).numpy().copy(order="K")
        opt = dataclasses.replace(self.basic_opt, lauum_par_blk_multiplier=1)

        act_lo = gpu_lauum(torch.from_numpy(mat), upper=True, overwrite=True, opt=opt)
        torch.cuda.synchronize()
        np.testing.assert_allclose(expected_upper, act_lo.numpy(), rtol=self.rtol[dtype])


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestLauumKernel:
    rtol = {np.float64: 1e-12, np.float32: 1e-5}

    @pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
    def test_lauum(self, dtype, get_mat, expected_lower):
        device = torch.device("cuda:0")

        mat = get_mat(order="F", dtype=dtype)
        gpu_in = move_tensor(mat, device)
        gpu_out = move_tensor(mat, device)
        gpu_out.fill_(0.0)

        # Run on the GPU
        cuda_lauum_lower(n=mat.shape[0], A=gpu_in, lda=gpu_in.stride(1), B=gpu_out, ldb=gpu_out.stride(1))
        torch.cuda.synchronize(device)

        # Compare outputs and print timing info
        np.testing.assert_allclose(np.tril(expected_lower), gpu_out.cpu().numpy(), rtol=self.rtol[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
    def test_strided(self, dtype, get_mat, expected_lower):
        device = torch.device("cuda:0")

        mat = get_mat(order="F", dtype=dtype)
        gpu_in = move_tensor(mat, device)
        gpu_in_strided = torch.cat([gpu_in, torch.zeros(gpu_in.shape[0], 10, device=device, dtype=gpu_in.dtype)], 1).T
        gpu_in_strided = gpu_in_strided[:gpu_in.shape[0], :gpu_in.shape[0]]
        gpu_in_strided.copy_(gpu_in)
        gpu_out = move_tensor(mat, device)
        gpu_out_strided = torch.cat([gpu_out, torch.zeros(gpu_out.shape[0], 10, device=device, dtype=gpu_in.dtype)], 1).T
        gpu_out_strided = gpu_out_strided[:gpu_out.shape[0], :gpu_out.shape[0]]
        gpu_out_strided.fill_(0.0)

        # Run on the GPU
        cuda_lauum_lower(n=gpu_in.shape[0], A=gpu_in_strided, lda=gpu_in_strided.stride(1), B=gpu_out_strided, ldb=gpu_out_strided.stride(1))
        torch.cuda.synchronize(device)

        # Compare outputs and print timing info
        np.testing.assert_allclose(np.tril(expected_lower), gpu_out_strided.cpu().numpy(), rtol=self.rtol[dtype])

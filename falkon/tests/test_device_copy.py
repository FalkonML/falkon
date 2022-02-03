import pytest
import numpy as np
import torch

from falkon import FalkonOptions
from falkon.utils import decide_cuda
from falkon.tests.conftest import fix_mat, memory_checker
from falkon.tests.gen_random import gen_random
from falkon.utils.device_copy import copy

n = 10_000
d = 1000


@pytest.fixture(scope="module")
def mat():
    return torch.from_numpy(gen_random(n, d, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def col_vec():
    return torch.from_numpy(gen_random(n, 1, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def row_vec():
    return torch.from_numpy(gen_random(1, n, 'float64', False, seed=92))


@pytest.fixture(scope="module")
def large_mat():
    return torch.from_numpy(gen_random(2*n, d, 'float64', False, seed=92))


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
def test_copy_host_to_dev(mat, order):
    in_mat: torch.Tensor = fix_mat(mat, np.float64, order, device="cpu", copy=True, numpy=False)
    output = torch.empty_strided(in_mat.size(), in_mat.stride(), dtype=in_mat.dtype, device="cuda")

    opt = FalkonOptions(max_gpu_mem=0.0)
    with memory_checker(opt):
        copy(in_mat, output)
    torch.testing.assert_allclose(in_mat, output.cpu(), rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
def test_copy_vec_host_to_dev(row_vec, col_vec, order):
    for test_vec in [row_vec, col_vec]:
        in_vec: torch.Tensor = fix_mat(test_vec, np.float64, order, device="cpu", copy=True, numpy=False)
        output = torch.empty_strided(in_vec.size(), in_vec.stride(), dtype=in_vec.dtype, device="cuda")
        opt = FalkonOptions(max_gpu_mem=0.0)
        with memory_checker(opt):
            copy(in_vec, output)
        torch.testing.assert_allclose(in_vec, output.cpu(), rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
def test_copy_dev_to_host(mat, order):
    in_mat: torch.Tensor = fix_mat(mat, np.float64, order, device="cuda", copy=True, numpy=False)
    output = torch.empty_strided(in_mat.size(), in_mat.stride(), dtype=in_mat.dtype, device="cpu")

    opt = FalkonOptions(max_gpu_mem=0.0)
    with memory_checker(opt):
        copy(in_mat, output)

    torch.testing.assert_allclose(in_mat.cpu(), output.cpu(), rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
def test_copy_vec_dev_to_host(row_vec, col_vec, order):
    for test_vec in [row_vec, col_vec]:
        in_vec: torch.Tensor = fix_mat(test_vec, np.float64, order, device="cuda", copy=True, numpy=False)
        output = torch.empty_strided(in_vec.size(), in_vec.stride(), dtype=in_vec.dtype, device="cpu")
        opt = FalkonOptions(max_gpu_mem=0.0)
        with memory_checker(opt):
            copy(in_vec, output)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("in_dev", ["cpu", "cuda"])
@pytest.mark.parametrize("size,out_size", [
    [(1, 100), (1, 100)],
    [(1, 100), (10, 200)],
    [(100, 1), (100, 1)],
    [(100, 1), (100, 10)],
    [(1, 1), (1, 1)],
    [(500, 100), (600, 100)]])
def test_diff_sizes(mat, order, in_dev, size, out_size):
    if in_dev == "cuda":
        out_dev = "cpu"
    else:
        out_dev = "cuda"
    in_mat: torch.Tensor = fix_mat(mat, np.float64, order=order, device=in_dev, copy=True, numpy=False)
    in_mat = in_mat[:size[0], :size[1]]

    if order == "F":
        output = torch.empty_strided(out_size, (1, out_size[0]), dtype=in_mat.dtype, device=out_dev)
    else:
        output = torch.empty_strided(out_size, (out_size[0], 1), dtype=in_mat.dtype, device=out_dev)
    output = output[:size[0], :size[1]]

    opt = FalkonOptions(max_gpu_mem=0.0)
    with memory_checker(opt):
        copy(in_mat, output)

    torch.testing.assert_allclose(in_mat.cpu(), output.cpu(), rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("in_dev", ["cpu", "cuda"])
def test_wrong_stride(mat, order, in_dev):
    if in_dev == "cuda":
        out_dev = "cpu"
    else:
        out_dev = "cuda"
    in_mat: torch.Tensor = fix_mat(mat, np.float64, order=order, device=in_dev, copy=True, numpy=False)

    if order == "F":  # Then output should be row-contiguous
        output = torch.empty_strided(in_mat.size(), (in_mat.shape[1], 1), dtype=in_mat.dtype, device=out_dev) #* 100.
    else:  # Then output should be col-contiguous
        output = torch.empty_strided(in_mat.size(), (1, in_mat.shape[0]), dtype=in_mat.dtype, device=out_dev) #* 100.

    opt = FalkonOptions(max_gpu_mem=0.0)
    with memory_checker(opt):
        with pytest.raises(ValueError):
            copy(in_mat, output)


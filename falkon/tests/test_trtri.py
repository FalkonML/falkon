import time
import pytest

import torch
import numpy as np
import scipy.linalg

from falkon.c_ext import trtri as my_trtri
from falkon.tests.conftest import fix_mat
from falkon.utils.tensor_helpers import move_tensor

num_points = 32 * 100


def gen_upper_tri(num_points):
    torch.manual_seed(10)
    w = torch.randn(num_points, 20)
    w = w @ w.T + torch.eye(num_points)
    return np.triu(w)


@pytest.fixture
def rtol():
    return {
        np.float64: 1e-12,
        np.float32: 1e-5,
    }

@pytest.fixture
def upper_tri():
    return gen_upper_tri(num_points)


@pytest.mark.parametrize("upper", [True, False], ids=["upper", "lower"])
@pytest.mark.parametrize("order", ["F", "C"])
def test_right_multiple(upper_tri, rtol, upper, order):
    dtype = np.float32
    if not upper:
        upper_tri = upper_tri.T
    mat = fix_mat(upper_tri, order=order, copy=True, numpy=True, dtype=dtype)

    t_s = time.time()
    exp = scipy.linalg.lapack.strtri(mat, lower=not upper, unitdiag=0, overwrite_c=0)[0]
    t_cpu = time.time() - t_s

    dev_mat = move_tensor(torch.from_numpy(mat), "cuda:0")
    t_s = time.time()
    out = my_trtri(dev_mat, lower=not upper, unitdiag=0)
    torch.cuda.synchronize()
    t_gpu = time.time() - t_s

    print("CPU time: %.2fs - GPU time: %.2fs" % (t_cpu, t_gpu))

    assert out.data_ptr() == dev_mat.data_ptr(), "CUDA-TRTRI not performed in-place."

    out_cpu = out.cpu().numpy()

    np.set_printoptions(precision=1, linewidth=200, threshold=10_000)
    #print("EXPECTED", exp)
    #print("ACTUAL", out_cpu)

    torch.testing.assert_allclose(torch.from_numpy(exp), torch.from_numpy(out_cpu), rtol=rtol[dtype], atol=rtol[dtype])
    #np.testing.assert_allclose(np.triu(exp), np.triu(out_cpu), rtol=rtol[dtype])


def test_small():
    dtype = np.float32
    mat = gen_upper_tri(4)
    mat = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 10]], dtype=torch.float32)
    mat = fix_mat(mat, order="F", copy=True, numpy=True, dtype=dtype)

    exp = scipy.linalg.lapack.strtri(mat, lower=0, unitdiag=0, overwrite_c=0)[0]

    dev_mat = move_tensor(torch.from_numpy(mat), "cuda:0")
    out = my_trtri(dev_mat, lower=False, unitdiag=False)

    np.set_printoptions(precision=3, linewidth=200, threshold=10_000)
    print()
    print("Expected\n", torch.from_numpy(exp).numpy())
    print("Actual\n", out.cpu().numpy())


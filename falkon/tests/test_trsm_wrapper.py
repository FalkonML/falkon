import tracemalloc

import numpy as np
import pytest
import torch
from scipy.linalg import blas as sclb
from torch.profiler import ProfilerActivity, profile, record_function

from falkon.la_helpers import trsm
from falkon.tests.conftest import fix_mat
from falkon.tests.gen_random import gen_random
from falkon.utils import decide_cuda
from falkon.utils.tensor_helpers import move_tensor

M = 100
T = 3


@pytest.fixture(scope="module")
def mat():
    A = gen_random(M, M, "float64", F=True, seed=10)
    return A @ A.T


@pytest.fixture(scope="module")
def arr():
    return gen_random(M, T, "float64", F=True, seed=12)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("transpose", [True, False], ids=["transpose", "no_transpose"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param("cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")]),
    ],
)
def test_trsm_wrapper(mat, arr, dtype, order, device, lower, transpose):
    rtol = 1e-2 if dtype == np.float32 else 1e-10
    alpha = 0.01

    n_mat = move_tensor(fix_mat(mat, dtype=dtype, order=order, copy=True), device=device)
    n_arr = move_tensor(fix_mat(arr, dtype=dtype, order=order, copy=True), device=device)
    expected = sclb.dtrsm(alpha, mat, arr, side=0, lower=lower, trans_a=transpose, overwrite_b=0)
    if device == "cuda":
        torch.cuda.synchronize()

    activity = [ProfilerActivity.CPU] if device == "cpu" else [ProfilerActivity.CUDA]
    with profile(activities=activity, profile_memory=True, record_shapes=True) as prof:
        actual = trsm(n_arr, n_mat, alpha=alpha, lower=lower, transpose=transpose)

    # Assert that the matrix n_mat is never copied within trsm.
    stats = prof.key_averages()
    for s in stats:
        assert s.cpu_memory_usage < (M * M * 4), f"{s.key} copies input matrix (CPU)"
        if s.device_memory_usage < (M * M * 9):
            # Relatively large allocation happen randomly? and we don't care about those
            assert s.device_memory_usage < (M * M * 4), f"{s.key} copies input matrix (CUDA)"

    np.testing.assert_allclose(expected, actual.cpu().numpy(), rtol=rtol)


if __name__ == "__main__":
    pytest.main()


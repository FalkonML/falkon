import numpy as np
import pytest
from scipy.linalg import blas as sclb

from falkon.la_helpers import trsm
from falkon.tests.conftest import fix_mat
from falkon.tests.gen_random import gen_random
from falkon.utils import decide_cuda
from falkon.utils.tensor_helpers import move_tensor

M = 50
T = 30


@pytest.fixture(scope="module")
def mat():
    return gen_random(M, M, "float64", F=True, seed=10)


@pytest.fixture(scope="module")
def arr():
    return gen_random(M, T, "float64", F=True, seed=12)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
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
    rtol = 1e-2 if dtype == np.float32 else 1e-11

    n_mat = move_tensor(fix_mat(mat, dtype=dtype, order=order, copy=True), device=device)
    n_arr = move_tensor(fix_mat(arr, dtype=dtype, order=order, copy=True), device=device)

    expected = sclb.dtrsm(1e-2, mat, arr, side=0, lower=lower, trans_a=transpose, overwrite_b=0)

    if device.startswith("cuda") and order == "C":
        with pytest.raises(ValueError):
            actual = trsm(n_arr, n_mat, alpha=1e-2, lower=lower, transpose=transpose)
    else:
        actual = trsm(n_arr, n_mat, alpha=1e-2, lower=lower, transpose=transpose)
        np.testing.assert_allclose(expected, actual.cpu().numpy(), rtol=rtol)


if __name__ == "__main__":
    pytest.main()

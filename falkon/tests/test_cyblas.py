import time
from functools import partial

import numpy as np
import pytest
from falkon.utils.cyblas import copy_triang, potrf, mul_triang, vec_mul_triang, zero_triang

from falkon.tests.conftest import fix_mat
from falkon.tests.gen_random import gen_random, gen_random_pd


class TestCopyTriang:
    t = 50

    @pytest.fixture(scope="class")
    def mat(self):
        return np.random.random((self.t, self.t))

    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_low(self, mat, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, numpy=True)
        mat_low = mat.copy(order="K")
        # Upper triangle of mat_low is 0
        mat_low[np.triu_indices(self.t, 1)] = 0
        copy_triang(mat_low, upper=False)

        assert np.sum(mat_low == 0) == 0
        np.testing.assert_array_equal(np.tril(mat), np.tril(mat_low))
        np.testing.assert_array_equal(np.triu(mat_low), np.tril(mat_low).T)
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_low))

        # Reset and try with `upper=True`
        mat_low[np.triu_indices(self.t, 1)] = 0
        copy_triang(mat_low, upper=True)  # Only the diagonal will be set.
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_low))

    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_up(self, mat, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, numpy=True)
        mat_up = mat.copy(order="K")
        # Upper triangle of mat_low is 0
        mat_up[np.tril_indices(self.t, -1)] = 0
        copy_triang(mat_up, upper=True)

        assert np.sum(mat_up == 0) == 0
        np.testing.assert_array_equal(np.triu(mat), np.triu(mat_up))
        np.testing.assert_array_equal(np.tril(mat_up), np.triu(mat_up).T)
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_up))

        # Reset and try with `upper=False`
        mat_up[np.tril_indices(self.t, -1)] = 0
        copy_triang(mat_up, upper=False)  # Only the diagonal will be set.
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_up))


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "dirty"])
@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "copy"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestPotrf:
    t = 50
    rtol = {
        np.float64: 1e-13,
        np.float32: 1e-6
    }

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random_pd(self.t, np.float64, F=True, seed=12345)

    @pytest.fixture(scope="class")
    def exp_lower(self, mat):
        return np.linalg.cholesky(mat)

    @pytest.fixture(scope="class")
    def exp_upper(self, exp_lower):
        return exp_lower.T

    def test_upper(self, mat, exp_upper, clean, overwrite, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, copy=False, numpy=True)
        inpt = mat.copy(order="K")

        our_chol = potrf(inpt, upper=True, clean=clean, overwrite=overwrite)
        if overwrite:
            assert inpt.ctypes.data == our_chol.ctypes.data, "Overwriting failed"

        if clean:
            np.testing.assert_allclose(exp_upper, our_chol, rtol=self.rtol[dtype])
            assert np.tril(our_chol, -1).sum() == 0
        else:
            np.testing.assert_allclose(exp_upper, np.triu(our_chol), rtol=self.rtol[dtype])
            np.testing.assert_allclose(np.tril(mat, -1), np.tril(our_chol, -1))

    def test_lower(self, mat, exp_lower, clean, overwrite, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, copy=False, numpy=True)
        inpt = mat.copy(order="K")

        our_chol = potrf(inpt, upper=False, clean=clean, overwrite=overwrite)
        if overwrite:
            assert inpt.ctypes.data == our_chol.ctypes.data, "Overwriting failed"

        if clean:
            np.testing.assert_allclose(exp_lower, our_chol, rtol=self.rtol[dtype])
            assert np.triu(our_chol, 1).sum() == 0
        else:
            np.testing.assert_allclose(exp_lower, np.tril(our_chol), rtol=self.rtol[dtype])
            np.testing.assert_allclose(np.triu(mat, 1), np.triu(our_chol, 1))


@pytest.mark.benchmark
def test_potrf_speed():
    t = 5000
    mat = gen_random_pd(t, np.float32, F=False, seed=12345)
    t_s = time.time()
    our_chol = potrf(mat, upper=False, clean=True, overwrite=False)
    our_time = time.time() - t_s

    t_s = time.time()
    np_chol = np.linalg.cholesky(mat)
    np_time = time.time() - t_s

    np.testing.assert_allclose(np_chol, our_chol, rtol=1e-5)
    print("Time for cholesky(%d): Numpy %.2fs - Our %.2fs" % (t, np_time, our_time))


class TestMulTriang:
    t = 500

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random(self.t, self.t, np.float64, F=False, seed=123)

    @pytest.mark.parametrize("preserve_diag", [True, False], ids=["preserve", "no-preserve"])
    @pytest.mark.parametrize("upper", [True, False], ids=["upper", "lower"])
    @pytest.mark.parametrize("order", ["F", "C"])
    def test_zero(self, mat, upper, preserve_diag, order):
        inpt1 = fix_mat(mat, dtype=mat.dtype, order=order, copy=True, numpy=True)
        inpt2 = inpt1.copy(order="K")

        k = 1 if preserve_diag else 0
        if upper:
            tri_fn = partial(np.triu, k=k)
        else:
            tri_fn = partial(np.tril, k=-k)

        mul_triang(inpt1, upper=upper, preserve_diag=preserve_diag, multiplier=0)
        assert np.sum(tri_fn(inpt1)) == 0

        if preserve_diag:
            zero_triang(inpt2, upper=upper)
            np.testing.assert_allclose(inpt1, inpt2)


class TestVecMulTriang:
    @pytest.fixture
    def mat(self):
        return np.array([[1, 1, 1],
                         [2, 2, 4],
                         [6, 6, 8]], dtype=np.float32)

    @pytest.fixture
    def vec(self):
        return np.array([0, 1, 0.5], dtype=np.float32)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_lower(self, mat, vec, order):
        mat = fix_mat(mat, order=order, dtype=mat.dtype, numpy=True, copy=True)

        out = vec_mul_triang(mat.copy(order="K"), upper=False, side=0, multiplier=vec)
        exp = np.array([[0, 1, 1], [2, 2, 4], [3, 3, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

        out = vec_mul_triang(mat.copy(order="K"), upper=False, side=1, multiplier=vec)
        exp = np.array([[0, 1, 1], [0, 2, 4], [0, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_upper(self, mat, vec, order):
        mat = fix_mat(mat, order=order, dtype=mat.dtype, numpy=True, copy=True)

        out = vec_mul_triang(mat.copy(order="K"), upper=True, side=0, multiplier=vec)
        exp = np.array([[0, 0, 0], [2, 2, 4], [6, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

        out = vec_mul_triang(mat.copy(order="K"), upper=True, side=1, multiplier=vec)
        exp = np.array([[0, 1, 0.5], [2, 2, 2], [6, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

    @pytest.mark.benchmark
    def test_large(self):
        t = 30_000
        mat = gen_random(t, t, np.float64, F=False, seed=123)
        vec = gen_random(t, 1, np.float64, F=False, seed=124).reshape((-1,))

        t_s = time.time()
        vec_mul_triang(mat, vec, upper=True, side=1)
        t_tri = time.time() - t_s

        t_s = time.time()
        mat *= vec
        t_full = time.time() - t_s

        print("Our took %.2fs -- Full took %.2fs" % (t_tri, t_full))

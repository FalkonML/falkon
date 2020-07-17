import numpy as np
import pytest
import torch

from falkon.center_selection import UniformSel
from falkon.tests.gen_random import gen_random, gen_sparse_matrix

M = 500
D = 20


@pytest.fixture
def rowmaj_arr() -> torch.Tensor:
    return torch.from_numpy(gen_random(M, D, 'float64', False))


@pytest.fixture
def colmaj_arr() -> torch.Tensor:
    return torch.from_numpy(gen_random(M, D, 'float64', True))


@pytest.fixture
def uniform_sel() -> UniformSel:
    return UniformSel(np.random.default_rng(0))


def test_c_order(uniform_sel, rowmaj_arr):
    centers = uniform_sel.select(rowmaj_arr, None, 100)
    assert centers.stride() == (D, 1), "UniformSel changed input stride"
    assert centers.size() == (100, D), "UniformSel did not output correct size"


def test_f_order(uniform_sel, colmaj_arr):
    centers = uniform_sel.select(colmaj_arr, None, 100)
    assert centers.stride() == (1, 100), "UniformSel changed input stride"
    assert centers.size() == (100, D), "UniformSel did not output correct size"


def test_great_m(uniform_sel, colmaj_arr):
    centers = uniform_sel.select(colmaj_arr, None, M + 1)
    assert centers.size() == (M, D), "UniformSel did not output correct size"


def test_sparse_csr(uniform_sel):
    sparse_csr = gen_sparse_matrix(M, D, np.float32, 0.01)
    centers = uniform_sel.select(sparse_csr, None, 100)
    assert centers.size() == (100, D), "UniformSel did not output correct size"
    assert centers.is_csr is True, "UniformSel did not preserve sparsity correctly"


def test_sparse_csc(uniform_sel):
    sparse_csc = gen_sparse_matrix(M, D, np.float32, 0.01).transpose_csc()
    centers = uniform_sel.select(sparse_csc, None, 5)
    assert centers.size() == (5, M), "UniformSel did not output correct size"
    assert centers.is_csc is True, "UniformSel did not preserve sparsity correctly"


def test_with_y(uniform_sel, colmaj_arr):
    Y = torch.empty(M, 1, dtype=colmaj_arr.dtype)
    centers, cY = uniform_sel.select(colmaj_arr, Y, 100)
    assert centers.stride() == (1, 100), "UniformSel changed input stride"
    assert centers.size() == (100, D), "UniformSel did not output correct size"
    assert cY.size() == (100, 1), "UniformSel did not output correct Y size"

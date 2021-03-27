import numpy as np
import pytest
import torch

import falkon
from falkon.center_selection import UniformSelector
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.utils import decide_cuda

M = 500
D = 20
num_centers = 100


@pytest.fixture
def rowmaj_arr() -> torch.Tensor:
    return torch.from_numpy(gen_random(M, D, 'float64', False))


@pytest.fixture
def colmaj_arr() -> torch.Tensor:
    return torch.from_numpy(gen_random(M, D, 'float64', True))


@pytest.fixture
def uniform_sel() -> UniformSelector:
    return UniformSelector(np.random.default_rng(0), num_centers=num_centers)


@pytest.mark.parametrize("device", [
    pytest.param("cpu"),
    pytest.param("cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])
])
def test_c_order(uniform_sel, rowmaj_arr, device):
    rowmaj_arr = rowmaj_arr.to(device=device)
    centers: torch.Tensor = uniform_sel.select(rowmaj_arr, None)
    assert centers.stride() == (D, 1), "UniformSel changed input stride"
    assert centers.size() == (num_centers, D), "UniformSel did not output correct size"
    assert centers.dtype == rowmaj_arr.dtype
    assert centers.device == rowmaj_arr.device


def test_cuda(uniform_sel, rowmaj_arr):
    centers: torch.Tensor = uniform_sel.select(rowmaj_arr, None)
    assert centers.stride() == (D, 1), "UniformSel changed input stride"
    assert centers.size() == (num_centers, D), "UniformSel did not output correct size"
    assert centers.dtype == rowmaj_arr.dtype
    assert centers.device == rowmaj_arr.device
    centers, idx = uniform_sel.select_indices(rowmaj_arr, None)
    assert centers.device == idx.device
    assert idx.dtype == torch.long
    assert len(idx) == num_centers


def test_f_order(uniform_sel, colmaj_arr):
    centers: torch.Tensor = uniform_sel.select(colmaj_arr, None)
    assert centers.stride() == (1, num_centers), "UniformSel changed input stride"
    assert centers.size() == (num_centers, D), "UniformSel did not output correct size"
    assert centers.dtype == colmaj_arr.dtype
    assert centers.device == colmaj_arr.device
    centers, idx = uniform_sel.select_indices(colmaj_arr, None)
    assert centers.device == idx.device
    assert idx.dtype == torch.long
    assert len(idx) == num_centers


def test_great_m(colmaj_arr):
    uniform_sel = UniformSelector(np.random.default_rng(0), num_centers=M + 1)
    centers: torch.Tensor = uniform_sel.select(colmaj_arr, None)
    assert centers.size() == (M, D), "UniformSel did not output correct size"
    assert centers.dtype == colmaj_arr.dtype
    assert centers.device == colmaj_arr.device
    centers, idx = uniform_sel.select_indices(colmaj_arr, None)
    assert centers.device == idx.device
    assert idx.dtype == torch.long
    assert len(idx) == centers.shape[0]


def test_sparse_csr(uniform_sel):
    sparse_csr = gen_sparse_matrix(M, D, np.float32, 0.01)
    centers: falkon.sparse.SparseTensor = uniform_sel.select(sparse_csr, None)
    assert centers.size() == (num_centers, D), "UniformSel did not output correct size"
    assert centers.is_csr is True, "UniformSel did not preserve sparsity correctly"
    assert centers.dtype == sparse_csr.dtype
    assert centers.device == sparse_csr.device
    centers, idx = uniform_sel.select_indices(sparse_csr, None)
    assert centers.device == idx.device
    assert idx.dtype == torch.long
    assert len(idx) == centers.shape[0]


def test_sparse_csc():
    uniform_sel = UniformSelector(np.random.default_rng(0), num_centers=5)
    sparse_csc = gen_sparse_matrix(M, D, np.float32, 0.01).transpose_csc()
    centers: falkon.sparse.SparseTensor = uniform_sel.select(sparse_csc, None)
    assert centers.size() == (5, M), "UniformSel did not output correct size"
    assert centers.is_csc is True, "UniformSel did not preserve sparsity correctly"
    assert centers.dtype == sparse_csc.dtype
    assert centers.device == sparse_csc.device
    centers, idx = uniform_sel.select_indices(sparse_csc, None)
    assert centers.device == idx.device
    assert idx.dtype == torch.long
    assert len(idx) == centers.shape[0]


def test_with_y(uniform_sel, colmaj_arr):
    Y = torch.empty(M, 1, dtype=colmaj_arr.dtype)
    centers, cY = uniform_sel.select(colmaj_arr, Y)
    assert centers.stride() == (1, num_centers), "UniformSel changed input stride"
    assert centers.size() == (num_centers, D), "UniformSel did not output correct size"
    assert cY.size() == (num_centers, 1), "UniformSel did not output correct Y size"
    assert centers.dtype == colmaj_arr.dtype
    assert centers.device == colmaj_arr.device
    assert cY.dtype == Y.dtype
    assert cY.device == Y.device
    centers, cY, idx = uniform_sel.select_indices(colmaj_arr, Y)
    assert idx.dtype == torch.long
    assert len(idx) == centers.shape[0]
    assert len(idx) == cY.shape[0]


if __name__ == "__main__":
    pytest.main(args=[__file__])
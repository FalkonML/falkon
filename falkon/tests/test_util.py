import numpy as np
import pytest
import scipy.sparse
import torch

import falkon.preconditioner.pc_utils
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.tests.gen_random import gen_random
from falkon.utils.helpers import check_same_dtype, sizeof_dtype, check_sparse


@pytest.mark.parametrize("F", [True, False], ids=["col-contig", "row-contig"])
def test_add_diag(F):
    A = torch.from_numpy(gen_random(1000, 1000, 'float64', F=F, seed=10))
    diag = 10**6
    falkon.preconditioner.pc_utils.inplace_add_diag_th(A, diag)
    assert torch.all((A.diagonal() > 10**5) & (A.diagonal() < 20**6))


def test_check_same_dtype_equal():
    smat = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 1]]).astype(np.float32))
    ts = [torch.tensor(0, dtype=torch.float32),
          SparseTensor.from_scipy(smat),
          None]
    assert check_same_dtype(*ts) is True


def test_check_same_dtype_empty():
    assert check_same_dtype() is True


def test_check_same_dtype_notequal():
    smat32 = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 1]]).astype(np.float32))
    smat64 = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 1]]).astype(np.float64))
    ts = [torch.tensor(0, dtype=torch.float32),
          torch.tensor(0, dtype=torch.float64),
          SparseTensor.from_scipy(smat32), ]
    assert check_same_dtype(*ts) is False

    ts = [torch.tensor(0, dtype=torch.float32),
          SparseTensor.from_scipy(smat32),
          SparseTensor.from_scipy(smat64), ]
    assert check_same_dtype(*ts) is False


def test_size_of_dtype():
    assert 8 == sizeof_dtype(np.float64)
    assert 4 == sizeof_dtype(np.float32)
    with pytest.raises(TypeError):
        sizeof_dtype(np.int32)

    assert 8 == sizeof_dtype(torch.float64)
    assert 4 == sizeof_dtype(torch.float32)
    with pytest.raises(TypeError):
        sizeof_dtype(torch.int32)


def test_check_sparse():
    smat = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 1]]).astype(np.float32))
    st = SparseTensor.from_scipy(smat)

    assert [False, True] == check_sparse(torch.tensor(0), st)
    assert [] == check_sparse()

import warnings
from typing import Tuple

import numpy as np
import pytest
import torch

from falkon.mkl_bindings.mkl_bind import mkl_lib
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.tests.gen_random import gen_sparse_matrix

_RTOL = {torch.float32: 1e-6, torch.float64: 1e-13}
try:
    mkl = mkl_lib()
except ImportError:
    warnings.warn("MKL not available. MKL tests will be skipped.")
    mkl = None


@pytest.fixture(scope="module")
def sparse1():
    A = gen_sparse_matrix(50, 100_000, np.float64, 1e-5)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="module")
def sparse2():
    B = gen_sparse_matrix(100_000, 50, np.float64, 1e-5)
    Bd = torch.from_numpy(B.to_scipy().todense())
    return B, Bd


def assert_sparse_equal(s1: SparseTensor, s2: SparseTensor):
    assert s1.sparse_type == s2.sparse_type, "Sparse types differ"
    np.testing.assert_equal(s1.indexptr.numpy(), s2.indexptr.numpy())
    np.testing.assert_equal(s1.index.numpy(), s2.index.numpy())
    np.testing.assert_allclose(s1.data.numpy(), s2.data.numpy())


@pytest.mark.skipif(mkl is None, reason="MKL not available.")
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=[pytest.mark.full()])], ids=["float32", "float64"]
)
def test_through_mkl(sparse1: Tuple[SparseTensor, torch.Tensor], dtype):
    orig, _ = sparse1
    orig = orig.to(dtype=dtype)
    mkl_sparse1 = mkl.mkl_create_sparse(orig)
    through = mkl.mkl_export_sparse(mkl_sparse1, orig.dtype, output_type="csr")
    assert_sparse_equal(orig, through)
    mkl.mkl_sparse_destroy(mkl_sparse1)


@pytest.mark.skipif(mkl is None, reason="MKL not available.")
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=[pytest.mark.full()])], ids=["float32", "float64"]
)
def test_through_mkl_scipy(sparse1: Tuple[SparseTensor, torch.Tensor], dtype):
    orig, _ = sparse1
    orig = orig.to(dtype=dtype)
    orig_scipy = orig.to_scipy()  # Needs to be in its own variable or will fail..
    mkl_sparse1 = mkl.mkl_create_sparse_from_scipy(orig_scipy)
    through = mkl.mkl_export_sparse(mkl_sparse1, orig.dtype, output_type="csr")
    assert_sparse_equal(orig, through)
    mkl.mkl_sparse_destroy(mkl_sparse1)


@pytest.mark.skipif(mkl is None, reason="MKL not available.")
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=[pytest.mark.full()])], ids=["float32", "float64"]
)
def test_convert_csr(sparse2: Tuple[SparseTensor, torch.Tensor], dtype):
    orig, dense = sparse2
    orig = orig.to(dtype=dtype)
    dense = dense.to(dtype=dtype)
    # Convert: M (CSR) -> M.T (CSC) -> M.T (CSR)
    orig_csc = orig.transpose_csc()
    mkl_csc = mkl.mkl_create_sparse(orig_csc)
    # Twice to check working of `destroy_original`
    mkl_csr = mkl.mkl_convert_csr(mkl_csc, destroy_original=False)
    mkl_csr2 = mkl.mkl_convert_csr(mkl_csc, destroy_original=True)

    csr = mkl.mkl_export_sparse(mkl_csr, orig.dtype, output_type="csr")
    np.testing.assert_allclose(dense.T, csr.to_scipy().todense())

    mkl.mkl_sparse_destroy(mkl_csr)
    mkl.mkl_sparse_destroy(mkl_csr2)


@pytest.mark.skipif(mkl is None, reason="MKL not available.")
@pytest.mark.skip(reason="Unknown MKL problems with large first dimension and creation of CSC.")
def test_csc_creation(sparse1: Tuple[SparseTensor, torch.Tensor]):
    # Note that this test works with sparse2 (e.g. see test_convert_csr)
    orig, dense = sparse1
    orig_csc = orig.transpose_csc()
    mkl_csc = mkl.mkl_create_sparse(orig_csc)
    mkl.mkl_sparse_destroy(mkl_csc)


@pytest.mark.skipif(mkl is None, reason="MKL not available.")
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=[pytest.mark.full()])], ids=["float32", "float64"]
)
def test_spmmd(sparse1, sparse2, dtype):
    # sparse1 @ sparse2
    smat1, dmat1 = sparse1
    smat1 = smat1.to(dtype=dtype)
    dmat1 = dmat1.to(dtype=dtype)
    smat2, dmat2 = sparse2
    smat2 = smat2.to(dtype=dtype)
    dmat2 = dmat2.to(dtype=dtype)
    dt = torch.float32

    outC = torch.zeros(smat1.shape[0], smat2.shape[1], dtype=dt)
    outF = torch.zeros(smat2.shape[1], smat1.shape[0], dtype=dt).T

    mkl = mkl_lib()
    mkl_As = mkl.mkl_create_sparse(smat1)
    mkl_Bs = mkl.mkl_create_sparse(smat2)
    mkl.mkl_spmmd(mkl_As, mkl_Bs, outC)
    mkl.mkl_spmmd(mkl_As, mkl_Bs, outF)
    expected = dmat1 @ dmat2
    np.testing.assert_allclose(expected, outC, rtol=_RTOL[dt])
    np.testing.assert_allclose(expected, outF, rtol=_RTOL[dt])

    mkl.mkl_sparse_destroy(mkl_As)
    mkl.mkl_sparse_destroy(mkl_Bs)


if __name__ == "__main__":
    pytest.main()

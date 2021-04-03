import pytest
import scipy.sparse

import torch
import numpy as np
from falkon.utils.tensor_helpers import create_fortran

from falkon.sparse import sparse_norm, sparse_square_norm, sparse_matmul
from falkon.utils import decide_cuda
from falkon.sparse.sparse_tensor import SparseTensor


@pytest.fixture(scope="module")
def csr_mat() -> SparseTensor:
    """
     -  2
     1  3
     4  -
    """
    indexptr = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    index = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    value = torch.tensor([2, 1, 3, 4], dtype=torch.float32)
    return SparseTensor(indexptr=indexptr, index=index, data=value, size=(3, 2), sparse_type="csr")


@pytest.fixture(scope="module")
def csc_mat() -> SparseTensor:
    indexptr = torch.tensor([0, 2, 3, 6], dtype=torch.long)
    index = torch.tensor([0, 2, 2, 0, 1, 2], dtype=torch.long)
    value = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    return SparseTensor(indexptr=indexptr, index=index, data=value, size=(3, 3), sparse_type="csc")


@pytest.mark.parametrize("function", [sparse_norm, sparse_square_norm])
class TestSparseNorm():
    def test_non_csr(self, csc_mat, function):
        with pytest.raises(RuntimeError) as exc_info:
            function(csc_mat, out=None)
        assert str(exc_info.value).endswith(
            "norm can only be applied on CSR tensors.")

    def test_different_dt(self, csr_mat, function):
        out = torch.empty(csr_mat.shape[0], dtype=torch.float64)
        with pytest.raises(ValueError) as exc_info:
            function(csr_mat, out=out)
        assert str(exc_info.value).startswith(
            "All data-types must match.")

    def test_wrong_out_shape(self, csr_mat, function):
        out = torch.empty(csr_mat.shape[0] + 1, dtype=torch.float32)
        with pytest.raises(ValueError) as exc_info:
            function(csr_mat, out=out)
        assert str(exc_info.value).startswith(
            "Dimension 0 of A must match the length of tensor 'out'.")

    def test_norm(self, csr_mat, function):
        exp_norm = np.linalg.norm(csr_mat.to_scipy(copy=True).todense(), axis=1).reshape(-1, 1)
        exp_square_norm = exp_norm ** 2
        if 'square' in function.__name__:
            exp = exp_square_norm
        else:
            exp = exp_norm

        act = function(csr_mat, out=None)
        torch.testing.assert_allclose(act, torch.from_numpy(exp).to(dtype=act.dtype))

    def test_norm_with_out(self, csr_mat, function):
        exp_norm = np.linalg.norm(csr_mat.to_scipy(copy=True).todense(), axis=1).reshape(-1, 1)
        exp_square_norm = exp_norm ** 2
        if 'square' in function.__name__:
            exp = exp_square_norm
        else:
            exp = exp_norm

        out = torch.empty(csr_mat.shape[0], 1, dtype=csr_mat.dtype, device=csr_mat.device)
        act = function(csr_mat, out=out)
        assert out.data_ptr() == act.data_ptr()
        torch.testing.assert_allclose(act, torch.from_numpy(exp).to(dtype=act.dtype))

        out = torch.empty(csr_mat.shape[0], dtype=csr_mat.dtype, device=csr_mat.device)
        act = function(csr_mat, out=out)
        assert out.data_ptr() == act.data_ptr()
        torch.testing.assert_allclose(act, torch.from_numpy(exp).to(dtype=act.dtype).reshape(-1))


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))
])
class TestMyTranspose():
    def test_simple_transpose(self, device, csr_mat):
        arr = csr_mat.to(device=device)
        tr_arr = arr.transpose_csc()
        assert tr_arr.shape == (2, 3), "expected transpose shape to be %s, but found %s" % ((2, 3), tr_arr.shape)
        tr_mat = tr_arr.to_scipy().tocoo()
        assert tr_mat.row.tolist() == [1, 0, 1, 0], "expected rows %s, but found %s" % ([1, 0, 1, 0], tr_mat.row.tolist())
        assert tr_mat.col.tolist() == [0, 1, 1, 2], "expected cols %s, but found %s" % ([0, 1, 1, 2], tr_mat.col.tolist())
        assert tr_mat.data.tolist() == [2, 1, 3, 4], "expected data %s, but found %s" % ([2, 1, 3, 4], tr_mat.data.tolist())


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))
])
class TestNarrow():
    def test_start_zero(self, device, csr_mat):
        arr = csr_mat.to(device=device)

        arr_small = arr.narrow_rows(0, 2)
        sm_coo = arr_small.to_scipy().tocoo()
        assert sm_coo.row.tolist() == [0, 1, 1]
        assert sm_coo.col.tolist() == [1, 0, 1]
        assert sm_coo.data.tolist() == [2, 1, 3]
        assert arr.indexptr.data_ptr() == arr_small.indexptr.data_ptr()

        arr_small = arr.narrow_rows(0, 1)
        sm_coo = arr_small.to_scipy().tocoo()
        assert sm_coo.row.tolist() == [0]
        assert sm_coo.col.tolist() == [1]
        assert sm_coo.data.tolist() == [2]
        assert arr.indexptr.data_ptr() == arr_small.indexptr.data_ptr()

    def test_start_mid(self, device, csr_mat):
        arr = csr_mat.to(device=device)

        arr_small = arr.narrow_rows(1, None)
        sm_coo = arr_small.to_scipy().tocoo()
        assert [0, 0, 1] == sm_coo.row.tolist()
        assert [0, 1, 0] == sm_coo.col.tolist()
        assert [1, 3, 4] == sm_coo.data.tolist()

        arr_small = arr.narrow_rows(1, 1)
        sm_coo = arr_small.to_scipy().tocoo()
        assert sm_coo.row.tolist() == [0, 0]
        assert sm_coo.col.tolist() == [0, 1]
        assert sm_coo.data.tolist() == [1, 3]

    def test_empty(self, device):
        indexptr = torch.tensor([0, 1, 1, 1, 3, 4], dtype=torch.long, device=device)
        index = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
        value = torch.tensor([2, 1, 3, 4], dtype=torch.float32, device=device)
        arr = SparseTensor(indexptr=indexptr, index=index, data=value, size=(5, 2), sparse_type="csr")

        arr_small = arr.narrow_rows(1, 2)
        sm_coo = arr_small.to_scipy().tocoo()
        assert sm_coo.row.tolist() == []
        assert sm_coo.col.tolist() == []
        assert sm_coo.data.tolist() == []


class TestMatMul():
    @pytest.fixture(scope="class")
    def mat1(self):
        return torch.randn(200, 10)

    @pytest.fixture(scope="class")
    def mat2(self):
        return torch.randn(10, 100)

    @pytest.fixture(scope="class")
    def expected(self, mat1, mat2):
        return mat1 @ mat2

    @pytest.mark.parametrize("device", [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))
    ])
    def test_matmul_zeros(self, mat1, mat2, expected, device):
        mat1_zero_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(torch.zeros_like(mat1).numpy())).to(device=device)
        mat2_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(mat2.numpy())).to(device=device)
        out = torch.empty_like(expected).to(device)
        sparse_matmul(mat1_zero_csr, mat2_csc, out)
        assert torch.all(out == 0.0)

        mat1_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat1.numpy())).to(device=device)
        mat2_zero_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(torch.zeros_like(mat2).numpy())).to(device=device)
        out = torch.empty_like(expected).to(device=device)
        sparse_matmul(mat1_csr, mat2_zero_csc, out)
        assert torch.all(out == 0.0)

    def test_cpu_matmul_wrong_format(self, mat1, mat2, expected):
        out = torch.empty_like(expected)
        mat1_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat1))
        mat2_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat2))
        with pytest.raises(ValueError) as exc_info:
            sparse_matmul(mat1_csr, mat2_csr, out)
        assert str(exc_info.value).startswith(
            "B must be CSC matrix")
        mat1_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(mat1))
        with pytest.raises(ValueError) as exc_info:
            sparse_matmul(mat1_csc, mat2_csr, out)
        assert str(exc_info.value).startswith(
            "A must be CSR matrix")

    def test_cpu_matmul(self, mat1, mat2, expected):
        out = torch.empty_like(expected)
        mat1_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat1))
        mat2_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(mat2))
        sparse_matmul(mat1_csr, mat2_csc, out)

        torch.testing.assert_allclose(out, expected)

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_matmul_wrong_format(self, mat1, mat2, expected):
        dev = torch.device("cuda:0")
        out = torch.empty_like(expected).to(device=dev)
        mat1_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat1)).to(device=dev)
        mat2_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(mat2)).to(device=dev)
        with pytest.raises(ValueError) as exc_info:
            sparse_matmul(mat1_csr, mat2_csc, out)
        assert str(exc_info.value).startswith(
            "B must be CSR matrix")
        mat1_csc = SparseTensor.from_scipy(scipy.sparse.csc_matrix(mat1))
        with pytest.raises(ValueError) as exc_info:
            sparse_matmul(mat1_csc, mat2_csc, out)
        assert str(exc_info.value).startswith(
            "A must be CSR matrix")

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_matmul(self, mat1, mat2, expected):
        dev = torch.device("cuda:0")
        out = create_fortran(expected.shape, expected.dtype, dev)
        mat1_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat1)).to(device=dev)
        mat2_csr = SparseTensor.from_scipy(scipy.sparse.csr_matrix(mat2)).to(device=dev)
        sparse_matmul(mat1_csr, mat2_csr, out)

        torch.testing.assert_allclose(out, expected)


if __name__ == "__main__":
    pytest.main()

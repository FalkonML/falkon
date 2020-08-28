import unittest
import torch
from falkon.sparse.sparse_tensor import SparseTensor


class TestMyTranspose(unittest.TestCase):
    def test_simple_transpose(self):
        for device in ('cpu', 'cuda:0'):
            with self.subTest(device=device):
                if device == 'cuda:0' and not torch.cuda.is_available():
                    self.skipTest("Cuda not available")

                indexptr = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
                index = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
                value = torch.tensor([2, 1, 3, 4], dtype=torch.float32, device=device)
                arr = SparseTensor(indexptr=indexptr, index=index, data=value, size=(3, 2), sparse_type="csr")
                tr_arr = arr.transpose_csc()
                self.assertEqual((2, 3), tr_arr.shape)
                tr_mat = tr_arr.to_scipy().tocoo()
                self.assertEqual(tr_mat.row.tolist(), [1, 0, 1, 0])
                self.assertEqual(tr_mat.col.tolist(), [0, 1, 1, 2])
                self.assertEqual(tr_mat.data.tolist(), [2, 1, 3, 4])


class TestNarrow(unittest.TestCase):
    def test_start_zero(self):
        device = 'cpu'
        indexptr = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
        index = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
        value = torch.tensor([2, 1, 3, 4], dtype=torch.float32, device=device)
        arr = SparseTensor(indexptr=indexptr, index=index, data=value, size=(3, 2), sparse_type="csr")

        arr_small = arr.narrow_rows(0, 2)
        sm_coo = arr_small.to_scipy().tocoo()
        self.assertEqual(sm_coo.row.tolist(), [0, 1, 1])
        self.assertEqual(sm_coo.col.tolist(), [1, 0, 1])
        self.assertEqual(sm_coo.data.tolist(), [2, 1, 3])
        self.assertEqual(arr.indexptr.data_ptr(), arr_small.indexptr.data_ptr())

        arr_small = arr.narrow_rows(0, 1)
        sm_coo = arr_small.to_scipy().tocoo()
        self.assertEqual(sm_coo.row.tolist(), [0])
        self.assertEqual(sm_coo.col.tolist(), [1])
        self.assertEqual(sm_coo.data.tolist(), [2])
        self.assertEqual(arr.indexptr.data_ptr(), arr_small.indexptr.data_ptr())

    def test_start_mid(self):
        device = 'cpu'
        indexptr = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
        index = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
        value = torch.tensor([2, 1, 3, 4], dtype=torch.float32, device=device)
        arr = SparseTensor(indexptr=indexptr, index=index, data=value, size=(3, 2), sparse_type="csr")

        arr_small = arr.narrow_rows(1, None)
        sm_coo = arr_small.to_scipy().tocoo()
        self.assertEqual([0, 0, 1], sm_coo.row.tolist())
        self.assertEqual([0, 1, 0], sm_coo.col.tolist())
        self.assertEqual([1, 3, 4], sm_coo.data.tolist())

        arr_small = arr.narrow_rows(1, 1)
        sm_coo = arr_small.to_scipy().tocoo()
        self.assertEqual(sm_coo.row.tolist(), [0, 0])
        self.assertEqual(sm_coo.col.tolist(), [0, 1])
        self.assertEqual(sm_coo.data.tolist(), [1, 3])

    def test_empty(self):
        device = 'cpu'
        indexptr = torch.tensor([0, 1, 1, 1, 3, 4], dtype=torch.long, device=device)
        index = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
        value = torch.tensor([2, 1, 3, 4], dtype=torch.float32, device=device)
        arr = SparseTensor(indexptr=indexptr, index=index, data=value, size=(5, 2), sparse_type="csr")

        arr_small = arr.narrow_rows(1, 2)
        sm_coo = arr_small.to_scipy().tocoo()
        self.assertEqual(sm_coo.row.tolist(), [])
        self.assertEqual(sm_coo.col.tolist(), [])
        self.assertEqual(sm_coo.data.tolist(), [])

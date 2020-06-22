from enum import Enum
from typing import Optional, Tuple, Union

import scipy.sparse
import torch


class SparseType(Enum):
    CSR = "csr"
    CSC = "csc"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


class SparseTensor():
    """Class to represent sparse 2D matrices

    `SparseTensor` can hold the data in CSR or CSC format, which represent the data with three
    1-dimensional arrays.
    It supports some of the common torch tensor management functions (e.g. `pin_memory`, `device`,
    `size`) and conversion to and from the corresponding scipy sparse matrix representation.
    It does **not** define any mathematical function on sparse matrices, which are
    instead defined in the `sparse_ops.py` file.

    Parameters
    ----------
    indexptr : torch.Tensor
        Array of row (or column for CSC data) pointers into the
        `index` and `data` arrays. Should be either of type long or int.
    index : torch.Tensor
        Array of column (or row for CSC data) indices for non-zero elements.
        Should be either of type long or int.
    data : torch.Tensor
        Array of the non-zero elements for the sparse matrix.
    size : Tuple[int, int]
        Shape of the 2D tensor (rows, columns).
    sparse_type: str or SparseType
        Whether the matrix should be interpreted as CSR or CSC format.
    """
    def __init__(self,
                 indexptr: torch.Tensor,
                 index: torch.Tensor,
                 data: torch.Tensor,
                 size: Tuple[int, int],
                 sparse_type: Union[str, SparseType] = SparseType.CSR):
        if isinstance(sparse_type, str):
            sparse_type = SparseType(sparse_type)
        if sparse_type == SparseType.CSR:
            if indexptr.size(0) - 1 != size[0]:
                raise ValueError("Data is not in correct csr format. Incorrect indexptr size.")
        elif sparse_type == SparseType.CSC:
            if indexptr.size(0) - 1 != size[1]:
                raise ValueError("Data is not in correct csc format. Incorrect indexptr size.")
        else:
            raise ValueError("Sparse type %s not valid." % (sparse_type))
        if index.size(0) != data.size(0):
            raise ValueError("Data is not in correct format. Different sizes for index and values.")

        dev = data.device
        if index.device != dev or indexptr.device != dev:
            raise ValueError("Cannot create SparseTensor with components on different devices.")

        self.indexptr = indexptr
        self.index = index
        self.data = data
        self.sparse_type = sparse_type
        self._size = size

    @property
    def shape(self):
        return self._size

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self._size
        return self._size[dim]

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def is_csc(self):
        return self.sparse_type == SparseType.CSC

    @property
    def is_csr(self):
        return self.sparse_type == SparseType.CSR

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def is_cuda(self) -> bool:
        return self.data.is_cuda

    def nnz(self):
        return self.data.numel()

    @property
    def density(self):
        return self.nnz() / (self._size[0] * self._size[1])

    def dim(self):
        return len(self._size)

    def narrow_rows(self, start: Optional[int], length: Optional[int]) -> 'SparseTensor':
        """Select a subset of contiguous rows from the sparse matrix.
        If this is a CSC sparse matrix, instead of taking contiguous rows we take contiguous
        columns.

        Parameters
        ----------
        start: int or None
            The index of the first row to select. If None will be assumed to be 0.
        length: int or None
            The number of rows to select. If None will be assumed to be all rows after `start`.

        Returns
        --------
        SparseTensor
            A new `SparseTensor` object with `length` rows.

        Notes
        ------
        The output matrix will share storage with the original matrix whenever possible.
        """
        if start is None:
            start = 0
        elif start > self.shape[0]:
            raise IndexError("Start is greater than the length of the array")
        if length is None:
            length = self.shape[0] - start
        elif length + start > self.shape[0]:
            raise IndexError("End larger than array")

        end = start + length
        startptr = self.indexptr[start]
        endptr = self.indexptr[end]

        new_indexptr = self.indexptr[start:end + 1]
        new_index = self.index[startptr:endptr]
        new_data = self.data[startptr:endptr]
        if start > 0:
            new_indexptr = new_indexptr.clone().detach()
            new_indexptr.sub_(startptr)  # subtract in place

        return SparseTensor(
            indexptr=new_indexptr, index=new_index, data=new_data, size=(length, self.size(1)))

    def to(self, dtype=None, device=None) -> 'SparseTensor':
        new_data = self.data
        new_indexptr = self.indexptr
        new_index = self.index
        
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        change_dtype = dtype != self.dtype
        change_device = device != self.device
        
        if change_dtype or change_device:
            new_data = self.data.to(dtype=dtype, device=device)
        if change_device:
            new_indexptr = self.indexptr.to(device=device)
            new_index = self.index.to(device=device)
        return SparseTensor(
            indexptr=new_indexptr, index=new_index, data=new_data, 
            size=self.shape, sparse_type=self.sparse_type)

    def index_to_int_(self):
        self.indexptr = self.indexptr.to(dtype=torch.int32)
        self.index = self.index.to(dtype=torch.int32)

    def index_to_int(self):
        new_index = self.index.to(dtype=torch.int32)
        new_indexptr = self.indexptr.to(dtype=torch.int32)
        return SparseTensor(
            indexptr=new_indexptr, index=new_index, data=self.data, 
            size=self.shape, sparse_type=self.sparse_type)

    def index_to_long_(self):
        self.indexptr = self.indexptr.to(dtype=torch.int64)
        self.index = self.index.to(dtype=torch.int64)

    def index_to(self, dtype: torch.dtype):
        new_index = self.index.to(dtype=dtype, copy=False)
        new_indexptr = self.indexptr.to(dtype=dtype, copy=False)
        return SparseTensor(
            indexptr=new_indexptr, index=new_index, data=self.data,
            size=self.shape, sparse_type=self.sparse_type)

    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.indexptr = self.indexptr.pin_memory()
        self.index = self.index.pin_memory()
        return self

    def transpose_csc(self):
        if self.is_csc:
            raise RuntimeError("Cannot transpose_csc since data is already in csc format")
        new_size = (self.shape[1], self.shape[0])
        return SparseTensor(
            indexptr=self.indexptr, index=self.index, data=self.data, size=new_size,
            sparse_type=SparseType.CSC)

    @staticmethod
    def from_scipy(mat: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]) -> 'SparseTensor':
        if isinstance(mat, scipy.sparse.csr_matrix):
            return SparseTensor(
                indexptr=torch.from_numpy(mat.indptr).to(torch.long),
                index=torch.from_numpy(mat.indices).to(torch.long),
                data=torch.from_numpy(mat.data),
                size=mat.shape[:2],
                sparse_type=SparseType.CSR)
        elif isinstance(mat, scipy.sparse.csc_matrix):
            return SparseTensor(
                indexptr=torch.from_numpy(mat.indptr).to(torch.long),
                index=torch.from_numpy(mat.indices).to(torch.long),
                data=torch.from_numpy(mat.data),
                size=mat.shape[:2],
                sparse_type=SparseType.CSC)
        else:
            raise NotImplementedError("Cannot convert type %s to SparseTensor. "
                                      "Please use the CSR or CSC formats" % (type(mat)))

    def to_scipy(self, copy: bool = False) -> Union[
            scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]:
        if self.is_cuda:
            return self.to(device="cpu").to_scipy(copy=copy)

        if self.is_csr:
            return scipy.sparse.csr_matrix((self.data, self.index, self.indexptr),
                                           shape=self.shape, copy=copy)
        elif self.is_csc:
            return scipy.sparse.csc_matrix((self.data, self.index, self.indexptr),
                                           shape=self.shape, copy=copy)
        else:
            raise NotImplementedError("Cannot convert %s matrix to scipy" % (self.sparse_type))

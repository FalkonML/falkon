import ctypes
from collections.abc import Callable
from typing import Union

import numpy as np
import scipy.sparse
import torch
from numpy.ctypeslib import as_array

from falkon.sparse.sparse_tensor import SparseTensor

__all__ = ("mkl_lib", "Mkl", "MklError")

__MKL = None
_scipy_sparse_type = Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
_sparse_mat_type = Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, SparseTensor]


def mkl_lib():
    global __MKL
    if __MKL is None:
        __MKL = Mkl()
    return __MKL


class MklSparseMatrix(ctypes.Structure):
    pass


class MklError(Exception):
    RETURN_CODES = {0: "SPARSE_STATUS_SUCCESS",
                    1: "SPARSE_STATUS_NOT_INITIALIZED",
                    2: "SPARSE_STATUS_ALLOC_FAILED",
                    3: "SPARSE_STATUS_INVALID_VALUE",
                    4: "SPARSE_STATUS_EXECUTION_FAILED",
                    5: "SPARSE_STATUS_INTERNAL_ERROR",
                    6: "SPARSE_STATUS_NOT_SUPPORTED"}

    def __init__(self, code, fn):
        self.code = code
        self.fn_name = fn.__name__
        self.str_code = MklError.RETURN_CODES.get(self.code, "Unknown Error")
        msg = f"MKL Error {self.code}: {self.fn_name} failed with error '{self.str_code}'"
        super().__init__(msg)


class Mkl():
    sparse_matrix_t = ctypes.POINTER(MklSparseMatrix)
    MKL_OPERATION_T = {
        'n': 10,
        'N': 10,
        't': 11,
        'T': 11,
    }
    MKL_ORDERING_T = {
        'C': 101,  # SPARSE_LAYOUT_ROW_MAJOR
        'c': 101,
        'F': 102,  # SPARSE_LAYOUT_COLUMN_MAJOR
        'f': 102,
    }

    @staticmethod
    def get_dtypes(num_bits):
        if num_bits == 32:
            return ctypes.c_int, np.int32, torch.int32
        elif num_bits == 64:
            return ctypes.c_longlong, np.int64, torch.int64
        else:
            raise ValueError("Dtype invalid.")

    @staticmethod
    def mkl_check_return_val(ret_val: int, fn_handle: Callable):
        if ret_val != 0:
            # noinspection PyTypeChecker
            raise MklError(ret_val, fn_handle)

    def __init__(self):
        self.libmkl = Mkl._load_mkl_lib()

        # Define dtypes empirically
        # Basically just try with int64s and if that doesn't work try with int32s
        # There's a way to do this with intel's mkl helper package but I don't want to add the dependency
        self.MKL_INT, self.NP_INT, self.TORCH_INT = Mkl.get_dtypes(64)
        self.define_mkl_interface()
        try:
            self.check_dtype()
        except (ValueError, OverflowError, MemoryError):
            self.MKL_INT, self.NP_INT, self.TORCH_INT = Mkl.get_dtypes(32)
            self.define_mkl_interface()
            try:
                self.check_dtype()
            except (ValueError, OverflowError, MemoryError) as err:
                raise ImportError("Unable to set MKL numeric type") from err

    def define_mkl_interface(self):
        self.libmkl.mkl_sparse_d_create_csr.restype = ctypes.c_int
        self.libmkl.mkl_sparse_d_create_csr.argtypes = [
            ctypes.POINTER(Mkl.sparse_matrix_t),
            ctypes.c_int,
            self.MKL_INT,
            self.MKL_INT,
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.c_double)]
        self.libmkl.mkl_sparse_s_create_csr.restype = ctypes.c_int
        self.libmkl.mkl_sparse_s_create_csr.argtypes = [
            ctypes.POINTER(Mkl.sparse_matrix_t),
            ctypes.c_int,
            self.MKL_INT,
            self.MKL_INT,
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.c_float)]
        self.libmkl.mkl_sparse_d_create_csc.restype = ctypes.c_int
        self.libmkl.mkl_sparse_d_create_csc.argtypes = [
            ctypes.POINTER(Mkl.sparse_matrix_t),
            ctypes.c_int,
            self.MKL_INT,
            self.MKL_INT,
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.c_double)]
        self.libmkl.mkl_sparse_s_create_csc.restype = ctypes.c_int
        self.libmkl.mkl_sparse_s_create_csc.argtypes = [
            ctypes.POINTER(Mkl.sparse_matrix_t),  # Output matrix
            ctypes.c_int,  # Indexing
            self.MKL_INT,
            self.MKL_INT,
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.c_float), ]

        self.libmkl.mkl_sparse_d_export_csr.restype = ctypes.c_int
        self.libmkl.mkl_sparse_d_export_csr.argtypes = [
            Mkl.sparse_matrix_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        ]
        self.libmkl.mkl_sparse_s_export_csr.restype = ctypes.c_int
        self.libmkl.mkl_sparse_s_export_csr.argtypes = [
            Mkl.sparse_matrix_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        ]
        self.libmkl.mkl_sparse_d_export_csc.restype = ctypes.c_int
        self.libmkl.mkl_sparse_d_export_csc.argtypes = [
            Mkl.sparse_matrix_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        ]
        self.libmkl.mkl_sparse_s_export_csc.restype = ctypes.c_int
        self.libmkl.mkl_sparse_s_export_csc.argtypes = [
            Mkl.sparse_matrix_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(self.MKL_INT),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(self.MKL_INT)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        ]

        self.libmkl.mkl_sparse_convert_csr.restype = ctypes.c_int
        self.libmkl.mkl_sparse_convert_csr.argtypes = [
            Mkl.sparse_matrix_t,
            ctypes.c_int,
            ctypes.POINTER(Mkl.sparse_matrix_t)
        ]

        self.libmkl.mkl_sparse_destroy.restype = ctypes.c_int
        self.libmkl.mkl_sparse_destroy.argtypes = [Mkl.sparse_matrix_t]

        self.libmkl.mkl_sparse_d_spmmd.restype = ctypes.c_int
        self.libmkl.mkl_sparse_d_spmmd.argtypes = [
            ctypes.c_int,  # operation (transpose, non-transpose, conjugate transpose)
            Mkl.sparse_matrix_t,  # A
            Mkl.sparse_matrix_t,  # B
            ctypes.c_int,  # layout of the dense matrix (column major or row major)
            ctypes.c_void_p,  # pointer to the output
            self.MKL_INT  # leading dimension of the output (ldC)
        ]
        self.libmkl.mkl_sparse_s_spmmd.restype = ctypes.c_int
        self.libmkl.mkl_sparse_s_spmmd.argtypes = [
            ctypes.c_int,  # operation (transpose, non-transpose, conjugate transpose)
            Mkl.sparse_matrix_t,  # A
            Mkl.sparse_matrix_t,  # B
            ctypes.c_int,  # layout of the dense matrix (column major or row major)
            ctypes.POINTER(ctypes.c_float),  # pointer to the output
            self.MKL_INT  # leading dimension of the output (ldC)
        ]

    @staticmethod
    def _load_mkl_lib():
        # https://github.com/flatironinstitute/sparse_dot/blob/a78252a6d671011cd0836a1427dd4853d3e239a5/sparse_dot_mkl/_mkl_interface.py#L244
        # Load mkl_spblas through the libmkl_rt common interface

        # ilp64 uses int64 indices, loading through libmkl_rt results in int32 indices
        # (for sparse matrices)
        MKL_SO_LINUX = ["libmkl_intel_ilp64.so", "libmkl_rt.so"]

        # There's probably a better way to do this
        libmkl, libmkl_loading_errors = None, []
        for so_file in MKL_SO_LINUX:
            try:
                libmkl = ctypes.cdll.LoadLibrary(so_file)
                break
            except (OSError, ImportError) as err:
                libmkl_loading_errors.append(err)

        if libmkl is None:
            ierr_msg = ("Unable to load the MKL libraries through either of %s. "
                        "Try setting $LD_LIBRARY_PATH.") % (MKL_SO_LINUX)
            ierr_msg += "\n\t" + "\n\t".join(map(lambda x: str(x), libmkl_loading_errors))
            raise ImportError(ierr_msg)
        return libmkl

    def check_dtype(self):
        """
        Test to make sure that this library works by creating a random sparse array in CSC format,
        then converting it to CSR format and making sure is has not raised an exception.
        """
        test_array: scipy.sparse.csc_matrix = scipy.sparse.random(
            5, 5, density=0.5, format="csr", dtype=np.float64, random_state=50)
        test_comparison = test_array.A
        csc_ref = None
        try:
            csc_ref = self.mkl_create_sparse_from_scipy(test_array)
            csr_ref = self.mkl_convert_csr(csc_ref)
            final_array = self.mkl_export_sparse(csr_ref, torch.float64, "csr")
            if not np.allclose(test_comparison, final_array.to_scipy().A):
                raise ValueError("Dtype is not valid.")
            self.mkl_sparse_destroy(csr_ref)
        finally:
            if csc_ref is not None:
                self.mkl_sparse_destroy(csc_ref)
        return True

    def mkl_create_sparse(self, mat: SparseTensor) -> sparse_matrix_t:
        """Create a MKL sparse matrix from a SparseTensor object

        The object created is an opaque container which can be passed to sparse MKL operations.
        The returned object needs to be manually freed using the :meth:`Mkl.mkl_sparse_destroy`
        method when it is not needed anymore.

        Parameters:
        -----------
         - mat : SparseTensor
            The input sparse tensor (can be in CSR or CSC format). This method only supports
            floating-point tensors (i.e. of types torch.float32 or torch.float64) which reside
            on the CPU.

        Returns:
        --------
        mkl_sparse_matrix
            The Mkl object representing the input tensor.

        Notes:
        ------
        Depending on the version of MKL  which is linked, the integer indices (i.e. two of the
        three arrays used in CSR/CSC sparse representations) may be copied and cast to the
        appropriate MKL integer type. To find out which integer type is used by the linked MKL
        version see :attr:`Mkl.TORCH_INT`.
        The floating-point data is never copied.
        """
        is_double = mat.dtype == torch.float64
        ctype = ctypes.c_double if is_double else ctypes.c_float
        rows = self.MKL_INT(mat.shape[0])
        cols = self.MKL_INT(mat.shape[1])
        if mat.is_csr:
            fn = self.libmkl.mkl_sparse_d_create_csr if is_double else self.libmkl.mkl_sparse_s_create_csr
        elif mat.is_csc:
            fn = self.libmkl.mkl_sparse_d_create_csc if is_double else self.libmkl.mkl_sparse_s_create_csc
        else:
            raise TypeError("Cannot create sparse matrix from unknown format")
        if len(mat.indexptr) == 0:
            raise RuntimeError("Input matrix is empty, cannot create MKL matrix.")

        # Make sure indices are of the correct integer type
        mat = self._check_index_typing(mat)
        # Load into a MKL data structure and check return
        ref = Mkl.sparse_matrix_t()
        ret_val = fn(ctypes.byref(ref),  # Output (by ref)
                     ctypes.c_int(0),  # 0-based indexing
                     rows,  # rows
                     cols,  # cols
                     ctypes.cast(mat.indexptr[:-1].data_ptr(), ctypes.POINTER(self.MKL_INT)),
                     ctypes.cast(mat.indexptr[1:].data_ptr(), ctypes.POINTER(self.MKL_INT)),
                     ctypes.cast(mat.index.data_ptr(), ctypes.POINTER(self.MKL_INT)),
                     ctypes.cast(mat.data.data_ptr(), ctypes.POINTER(ctype)))  # values
        Mkl.mkl_check_return_val(ret_val, fn)
        return ref

    def mkl_create_sparse_from_scipy(self, matrix: _scipy_sparse_type) -> sparse_matrix_t:
        """Create a MKL sparse matrix from a scipy sparse matrix.

        The object created is an opaque container which can be passed to sparse MKL operations.
        The returned object needs to be manually freed using the :meth:`Mkl.mkl_sparse_destroy`
        method when it is not needed anymore.

        Parameters:
        -----------
         - mat : scipy.sparse.spmatrix
            The input sparse matrix which can be either in CSR or in CSC format. If any other
            format is used this method will throw an error. This method only supports
            floating-point tensors (i.e. of types torch.float32 or torch.float64).

        Returns:
        --------
        mkl_sparse_matrix
            The Mkl object representing the input tensor.

        Notes:
        ------
        Depending on the version of MKL  which is linked, the integer indices (i.e. two of the
        three arrays used in CSR/CSC sparse representations) may be copied and cast to the
        appropriate MKL integer type (the input object *will be modified*).
        To find out which integer type is used by the linked MKL version see :attr:`Mkl.TORCH_INT`.
        The floating-point data is never copied.
        """
        # Figure out which dtype for data
        if matrix.dtype == np.float32:
            double_precision = False
            ctype = ctypes.c_float
        elif matrix.dtype == np.float64:
            double_precision = True
            ctype = ctypes.c_double
        else:
            raise ValueError("Only float32 or float64 dtypes are supported")
        if len(matrix.indices) == 0:
            raise RuntimeError("Input matrix is empty, cannot create MKL matrix.")
        # Figure out which matrix creation function to use
        if scipy.sparse.isspmatrix_csr(matrix):
            assert matrix.indptr.shape[0] == matrix.shape[0] + 1
            fn = self.libmkl.mkl_sparse_d_create_csr if double_precision else self.libmkl.mkl_sparse_s_create_csr
        elif scipy.sparse.isspmatrix_csc(matrix):
            assert matrix.indptr.shape[0] == matrix.shape[1] + 1
            fn = self.libmkl.mkl_sparse_d_create_csc if double_precision else self.libmkl.mkl_sparse_s_create_csc
        else:
            raise ValueError("Matrix is not CSC or CSR")

        # Make sure indices are of the correct integer type
        matrix = self._check_index_typing(matrix)
        ref = Mkl.sparse_matrix_t()
        # Load into a MKL data structure and check return
        ret_val = fn(ctypes.byref(ref),
                     ctypes.c_int(0),
                     self.MKL_INT(matrix.shape[0]),
                     self.MKL_INT(matrix.shape[1]),
                     matrix.indptr.ctypes.data_as(ctypes.POINTER(self.MKL_INT)),
                     matrix.indptr[1:].ctypes.data_as(ctypes.POINTER(self.MKL_INT)),
                     matrix.indices.ctypes.data_as(ctypes.POINTER(self.MKL_INT)),
                     matrix.data.ctypes.data_as(ctypes.POINTER(ctype)))
        Mkl.mkl_check_return_val(ret_val, fn)
        return ref

    def _check_index_typing(self, sparse_matrix: _sparse_mat_type):
        int_max = np.iinfo(self.NP_INT).max

        if isinstance(sparse_matrix, SparseTensor):
            nnz = sparse_matrix.nnz()
        else:
            nnz = sparse_matrix.nnz

        if (nnz > int_max) or (max(sparse_matrix.shape) > int_max):
            msg = "MKL interface is {t} and cannot hold matrix {m}".format(
                m=repr(sparse_matrix), t=self.NP_INT)
            raise ValueError(msg)

        # Cast indexes to MKL_INT type
        if isinstance(sparse_matrix, SparseTensor):
            return sparse_matrix.index_to(dtype=self.TORCH_INT)
        else:
            if sparse_matrix.indptr.dtype != self.NP_INT:
                sparse_matrix.indptr = sparse_matrix.indptr.astype(self.NP_INT)
            if sparse_matrix.indices.dtype != self.NP_INT:
                sparse_matrix.indices = sparse_matrix.indices.astype(self.NP_INT)
            return sparse_matrix

    def mkl_export_sparse(self,
                          mkl_mat: sparse_matrix_t,
                          dtype: torch.dtype,
                          output_type: str = "csr") -> SparseTensor:
        """Create a :class:`SparseTensor` from a MKL sparse matrix holder.

        Note that not all possible MKL sparse matrices are supported (for example if 1-based
        indexing is used, or for non floating-point types), but those created with
        :meth:`mkl_create_sparse_from_scipy` and :meth:`mkl_create_sparse` are.

        Parameters:
        -----------
         - mkl_mat
            The MKL sparse matrix holder
         - dtype
            The data-type of the matrix. This must match the data-type of the data stored in
            the MKL matrix (no type conversion is performed), otherwise garbage data or memory
            corruption could occur.
         - output_type
            Whether the matrix should be interpreted as CSR (pass ``"csr"``) or CSC
            (pass ``"csc"``). This should match the MKL matrix, otherwise a transposed output
            may be produced.

        Returns:
        --------
        The :class:`SparseTensor` object, sharing the same data arrays as the MKL matrix.

        Notes:
        ------
        Depending on the integer type of the linked MKL version, the indices of the matrix may
        be copied. In any case the output tensor will use :class:`torch.int64` indices.
        """
        indptrb = ctypes.POINTER(self.MKL_INT)()
        indptren = ctypes.POINTER(self.MKL_INT)()
        indices = ctypes.POINTER(self.MKL_INT)()

        ordering = ctypes.c_int()
        nrows = self.MKL_INT()
        ncols = self.MKL_INT()

        if output_type.lower() == "csr":
            if dtype == torch.float64:
                fn = self.libmkl.mkl_sparse_d_export_csr
                ctype = ctypes.c_double
            elif dtype == torch.float32:
                fn = self.libmkl.mkl_sparse_s_export_csr
                ctype = ctypes.c_float
            else:
                raise TypeError("Data type %s not valid to export" % (dtype))
        elif output_type.lower() == "csc":
            if dtype == torch.float64:
                fn = self.libmkl.mkl_sparse_d_export_csc
                ctype = ctypes.c_double
            elif dtype == torch.float32:
                fn = self.libmkl.mkl_sparse_s_export_csc
                ctype = ctypes.c_float
            else:
                raise TypeError("Data type %s not valid to export" % (dtype))
        else:
            raise ValueError("Output type %s not valid" % (output_type))

        data_ptr = ctypes.POINTER(ctype)()

        ret_val = fn(mkl_mat,
                     ctypes.byref(ordering),
                     ctypes.byref(nrows),
                     ctypes.byref(ncols),
                     ctypes.byref(indptrb),
                     ctypes.byref(indptren),
                     ctypes.byref(indices),
                     ctypes.byref(data_ptr))
        Mkl.mkl_check_return_val(ret_val, fn)

        if ordering.value != 0:
            raise ValueError("1-based indexing (F-style) is not supported")
        ncols = ncols.value
        nrows = nrows.value

        # Get the index dimension
        index_dim = nrows if output_type == "csr" else ncols
        # Construct a numpy array and add 0 to first position for scipy.sparse's 3-array indexing
        indptrb = as_array(indptrb, shape=(index_dim,))
        indptren = as_array(indptren, shape=(index_dim,))

        indptren = np.insert(indptren, 0, indptrb[0])
        nnz = indptren[-1] - indptrb[0]

        # Construct numpy arrays from data pointer and from indicies pointer
        data = np.array(as_array(data_ptr, shape=(nnz,)), copy=True)
        indices = np.array(as_array(indices, shape=(nnz,)), copy=True)

        return SparseTensor(indexptr=torch.from_numpy(indptren).to(torch.long),
                            index=torch.from_numpy(indices).to(torch.long),
                            data=torch.from_numpy(data),
                            size=(nrows, ncols),
                            sparse_type=output_type.lower())

    def mkl_convert_csr(self, mkl_mat: sparse_matrix_t, destroy_original=False) -> sparse_matrix_t:
        """Convert a MKL matrix from CSC format to CSR format.

        Parameters:
        ----------
        mkl_mat
            The input, CSC format, MKL matrix.
        destroy_original
            Whether the input matrix will be freed (by calling :meth:`mkl_sparse_destroy`)
            after conversion.

        Returns:
        --------
        csr_mat
            Converted CSR MKL matrix.
        """
        csr_ref = Mkl.sparse_matrix_t()
        ret_val = self.libmkl.mkl_sparse_convert_csr(mkl_mat, Mkl.MKL_OPERATION_T['N'],
                                                     ctypes.byref(csr_ref))

        try:
            Mkl.mkl_check_return_val(ret_val, self.libmkl.mkl_sparse_convert_csr)
        except MklError:
            try:
                self.mkl_sparse_destroy(csr_ref)
            except ValueError:
                pass
            raise

        if destroy_original:
            self.mkl_sparse_destroy(mkl_mat)

        return csr_ref

    def mkl_sparse_destroy(self, ref_handle: sparse_matrix_t):
        """Free memory used by a MKL sparse matrix object.

        Note that this does not free the memory allocated to keep the data of the matrix itself,
        but only the sparse-matrix datastructure.

        Parameters:
        -----------
        ref_handle
            The MKL sparse matrix handle
        """
        ret_val = self.libmkl.mkl_sparse_destroy(ref_handle)
        Mkl.mkl_check_return_val(ret_val, self.libmkl.mkl_sparse_destroy)

    def mkl_spmmd(self,
                  A: sparse_matrix_t,
                  B: sparse_matrix_t,
                  out: torch.Tensor,
                  transposeA: bool = False) -> None:
        """Multiply two sparse matrices, storing the result in a dense matrix.

        This function wraps the ``mkl_sparse_?_spmmd`` functions from the MKL library (see MKL_),
        and is only implemented for float32 or float64 data-types.
        The computation performed is :math:`A @ B` or :math:`A.T @ B` depending on the value
        of `transposeA`.

        Both `A` and `B` must be in the same (CSR or CSC) sparse format, otherwise MKL will
        complain.

        Parameters:
        -----------
        A
            Input (left hand side) MKL sparse matrix.
        B
            Input (right hand side) MKL sparse matrix.
        out
            Output dense tensor. The result of the matrix multiplication will be stored here.
            The data-type of `out` must match that of both `A` and `B`, otherwise the operation
            will either produce garbage results or crash.
        transposeA
            Whether the left hand side matrix should be transposed.

        Examples:
        ---------
        Multiply two 2x2 sparse matrices
        >>> A = SparseTensor(torch.LongTensor([0, 2, 2]), torch.LongTensor([0, 1]),
        ...                  torch.Tensor([9.2, 6.3]), size=(2, 2), sparse_type="csr")
        >>> B = SparseTensor(torch.LongTensor([0, 1, 2]), torch.LongTensor([0, 1]),
        ...                  torch.Tensor([0.5, 1.0]), size=(2, 2), sparse_type="csr")
        >>> mkl = mkl_lib()
        >>> A_mkl = mkl.mkl_create_sparse(A)
        >>> B_mkl = mkl.mkl_create_sparse(B)
        >>> out = torch.empty(2, 2, dtype=torch.float32)
        >>> mkl.mkl_spmmd(A_mkl, B_mkl, out, transposeA=False)
        >>> out
        tensor([[4.6000, 6.3000],
                [0.0000, 0.0000]])
        >>> mkl.mkl_sparse_destroy(A_mkl)
        >>> mkl.mkl_sparse_destroy(B_mkl)

        .. _MKL: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/inspector-executor-sparse-blas-execution-routines/mkl-sparse-spmmd.html
        """
        if out.stride(0) == 1:  # Then it's column-contiguous (Fortran)
            layout = Mkl.MKL_ORDERING_T['F']
            ldC = out.stride(1)
        elif out.stride(1) == 1:
            layout = Mkl.MKL_ORDERING_T['C']
            ldC = out.stride(0)
        else:
            raise ValueError("Output matrix 'out' is not memory-contiguous")

        if transposeA:
            op = Mkl.MKL_OPERATION_T['T']
        else:
            op = Mkl.MKL_OPERATION_T['N']

        if out.dtype == torch.float32:
            fn = self.libmkl.mkl_sparse_s_spmmd
            output_ctype = ctypes.c_float
        elif out.dtype == torch.float64:
            fn = self.libmkl.mkl_sparse_d_spmmd
            output_ctype = ctypes.c_double
        else:
            raise TypeError("Data type %s not valid for SPMMD" % (out.dtype))

        out_ptr = out.data_ptr()
        out_ptr = ctypes.cast(out_ptr, ctypes.POINTER(output_ctype))
        ret_val = fn(op,  # Transpose or not
                     A,  # Output of _create_mkl_sparse(A)
                     B,  # Output of _create_mkl_sparse(B)
                     layout,  # Fortran or C layout
                     out_ptr,  # Pointer to the output
                     self.MKL_INT(ldC)  # Output leading dimension
                     )
        Mkl.mkl_check_return_val(ret_val, fn)

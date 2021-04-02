# cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_lapack

"""
Compilation instructions

$ cython -a cyblas.pyx
$ gcc -shared -pthread -fopenmp -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
      -I$CONDA_ENV_ROOT/include/python3.7m \
      -I$CONDA_ENV_ROOT/lib/python3.7/site-packages/numpy/core/include \
      -o cyblas.so cyblas.c
"""

class BlasError(Exception):
    pass

ctypedef fused float_type:
    np.float64_t
    np.float32_t



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def vec_mul_triang(np.ndarray[float_type, ndim=2] array,
                   np.ndarray[float_type, ndim=1] multiplier,
                   bint upper,
                   int side):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input matrix to vec_mul_triang must be square.")
    if cols != multiplier.shape[0]:
        raise ValueError("Multiplier shape mismatch. Expected %d found %d" %
                         (cols, multiplier.shape[0]))

    cdef float_type mul
    cdef int i, j
    if array.flags.f_contiguous: # Column-contiguous
        if upper and side == 1:  # 1 - upper=1, side=1
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                mul = multiplier[j]
                for i in range(j + 1):
                    array[i, j] *= mul
        elif upper and side == 0:  # 2 - upper=1, side=0
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                for i in range(j + 1):
                    mul = multiplier[i]
                    array[i, j] *= mul
        elif side == 1:  # 3 - upper=0, side=1
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                mul = multiplier[j]
                for i in range(j, rows):
                    array[i, j] *= mul
        else: # 4 - upper=0, side=0
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                for i in range(j, rows):
                    mul = multiplier[i]
                    array[i, j] *= mul
    elif array.flags.c_contiguous:
        if upper and side == 1:  # 4 - upper=1, side=1
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                for j in range(i, cols):
                    mul = multiplier[j]
                    array[i, j] *= mul
        elif upper and side == 0:  # 3 - upper=1, side=0
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                mul = multiplier[i]
                for j in range(i, cols):
                    array[i, j] *= mul
        elif side == 1:  # 2 - upper=0, side=1
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                for j in range(i + 1):
                    mul = multiplier[j]
                    array[i, j] *= mul
        else:  # 1 - upper=0, side=0
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                mul = multiplier[i]
                for j in range(i + 1):
                    array[i, j] *= mul
    else:
        raise ValueError("Matrix is not memory-contiguous")

    return array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mul_triang(np.ndarray[float_type, ndim=2] array,
               bint upper,
               bint preserve_diag,
               float_type multiplier):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    cdef int info = 0

    if cols != rows:
        raise ValueError("Input matrix to lascl must be square.")

    cdef char arr_type
    if array.flags.f_contiguous:
        arr_type = b'U' if upper else b'L'
    elif array.flags.c_contiguous:
        arr_type = b'L' if upper else b'U'
    else:
        raise MemoryError("Array is not contiguous.")

    # KL, KU are not used
    cdef int KL = 0
    cdef int KU = 0

    cdef float_type CFROM = 1
    cdef float_type CTO = multiplier

    cdef np.ndarray[float_type, ndim=1] diag
    if preserve_diag:
        diag = np.diagonal(array).copy()

    if float_type is np.float64_t:
        scipy.linalg.cython_lapack.dlascl(&arr_type, &KL, &KU, &CFROM, &CTO, &rows, &cols, &array[0,0], &rows, &info)
    elif float_type is np.float32_t:
        scipy.linalg.cython_lapack.slascl(&arr_type, &KL, &KU, &CFROM, &CTO, &rows, &cols, &array[0,0], &rows, &info)
    if info != 0:
        raise BlasError("LAPACK lascl failed with status %s" % (str(info)))

    if preserve_diag:
        np.fill_diagonal(array, diag)

    return array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def potrf(np.ndarray[float_type, ndim=2] array, bint upper, bint clean, bint overwrite):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    cdef int info = 0
    if cols != rows:
        raise ValueError("Input matrix to potrf must be square.")

    cdef char arr_type
    if array.flags.f_contiguous:
        arr_type = b'U' if upper else b'L'
    elif array.flags.c_contiguous:
        arr_type = b'L' if upper else b'U'
    else:
        raise MemoryError("Array is not contiguous.")

    # Copy array if necessary
    if not overwrite:
        array = array.copy(order='A')

    # Run Cholesky Factorization
    if float_type is np.float64_t:
        scipy.linalg.cython_lapack.dpotrf(&arr_type, &rows, &array[0, 0], &rows, &info)
    elif float_type is np.float32_t:
        scipy.linalg.cython_lapack.spotrf(&arr_type, &rows, &array[0, 0], &rows, &info)
    if info != 0:
        raise BlasError(
            "LAPACK potrf failed with status %s. Params: uplo %s , rows %d" %
            (str(info), str(arr_type), rows))

    # Clean non-factorized part of the matrix
    if clean:
        mul_triang(array, not upper, True, 0.0)

    # Transpose the matrix if it is C-contig.
    # if array.flags.c_contiguous:
    #     array = array.T

    return array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def copy_triang(np.ndarray[float_type, ndim=2] array, bint upper):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input array to copy_triang must be square.")

    cdef bint fin_upper = upper
    cdef bint transpose = False
    if array.flags.f_contiguous:
        upper = not upper
        array = array.T
        transpose = True

    cdef int i, j
    if upper:
        for i in prange(cols, nogil=True, schedule='guided', chunksize=max(rows//1000, 1)):
            for j in range(0, i):
                array[i, j] = array[j, i]
    else:
        for i in prange(cols - 1, -1, -1, nogil=True, schedule='guided', chunksize=max(rows//1000, 1)):
            for j in range(i+1, rows):
                array[i, j] = array[j, i]

    if transpose:
        return array.T
    return array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def add_symmetrize(np.ndarray[float_type, ndim=2] array):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input array to copy_triang must be square.")

    cdef bint transpose = False
    if array.flags.f_contiguous:
        array = array.T
        transpose = True

    cdef int i, j
    cdef float_type temp
    for i in prange(cols, nogil=True, schedule='guided', chunksize=max(rows//1000, 1)):
        for j in range(0, i):
            temp = array[i, j]
            array[i, j] += array[j, i]
            array[j, i] += temp
        array[i, i] *= 2

    if transpose:
        return array.T
    return array


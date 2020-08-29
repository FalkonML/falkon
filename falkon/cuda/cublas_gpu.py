import sys
from ctypes import (c_int, c_void_p)

import ctypes

__all__ = ("cublasSetMatrix", "cublasSetMatrixAsync",
           "cublasGetMatrix", "cublasGetMatrixAsync",
           "cublasGetStream", "cublasSetStream",
           "cublasDsyrk", "cublasSsyrk",
           "cublasDgemm", "cublasSgemm",
           "cublasDtrsm", "cublasStrsm",
           "cublasDtrsmBatched", "cublasStrsmBatched",
           "cublasXtStrsm", "cublasXtDtrsm",
           "cublasXtGetBlockDim", "cublasXtSetBlockDim",
           "cublasDtrmm", "cublasStrmm",
           "cublasCreate", "cublasDestroy",
           )

# Global variables
CUBLAS_EXCEPTIONS = {
    1: "cublasNotInitialized",
    3: "cublasAllocFailed",
    7: "cublasInvalidValue",
    8: "cublasArchMismatch",
    11: "cublasMappingError",
    13: "cublasExecutionFailed",
    14: "cublasInternalError",
    15: "cublasNotSupported",
    16: "cublasLicenseError"
}

CUBLAS_OP = {
    0: 0,  # CUBLAS_OP_N
    'n': 0,
    'N': 0,
    1: 1,  # CUBLAS_OP_T
    't': 1,
    'T': 1,
    2: 2,  # CUBLAS_OP_C
    'c': 2,
    'C': 2,
}
CUBLAS_FILL_MODE = {
    0: 0,  # CUBLAS_FILL_MODE_LOWER
    'l': 0,
    'L': 0,
    1: 1,  # CUBLAS_FILL_MODE_UPPER
    'u': 1,
    'U': 1,
}
CUBLAS_DIAG = {
    0: 0,  # CUBLAS_DIAG_NON_UNIT,
    'n': 0,
    'N': 0,
    1: 1,  # CUBLAS_DIAG_UNIT
    'u': 1,
    'U': 1,
}
CUBLAS_SIDE_MODE = {
    0: 0,  # CUBLAS_SIDE_LEFT
    'l': 0,
    'L': 0,
    1: 1,  # CUBLAS_SIDE_RIGHT
    'r': 1,
    'R': 1
}


# Error handling
class CublasError(Exception):
    """CUBLAS error"""

    def __init__(self, status_code):
        self.status_code = status_code
        try:
            self.message = CUBLAS_EXCEPTIONS[status_code]
        except KeyError:
            self.message = "Unknown CuBLAS error %d" % (status_code)
        super.__init__(self.message)


def cublas_check_status(status):
    """
    Raise CUBLAS exception
    Raise an exception corresponding to the specified CUBLAS error
    code.
    Parameters
    ----------
    status : int
        CUBLAS error code.
    See Also
    --------
    cublasExceptions
    """
    if status != 0:
        raise CublasError(status)


# Library Loading
def load_cublas_library():
    linux_version_list = [10.2, 10.1, 10.0, 9.2, 9.1, 9.0]
    if 'linux' in sys.platform:
        libcublas_libname_list = ['libcublas.so'] + \
                                 ['libcublas.so.%s' % v for v in linux_version_list]
    elif sys.platform == 'darwin':
        libcublas_libname_list = ['libcublas.dylib']
    else:
        raise RuntimeError('unsupported platform %s' % (sys.platform))

    # Print understandable error message when library cannot be found:
    libcublas = None
    for libcublas_libname in libcublas_libname_list:
        try:
            libcublas = ctypes.cdll.LoadLibrary(libcublas_libname)
        except OSError:
            pass
        else:
            break
    if libcublas is None:
        raise OSError('cublas library not found')
    return libcublas


_libcublas = load_cublas_library()


# Context Creation/Destruction
_libcublas.cublasCreate_v2.restype = int
_libcublas.cublasCreate_v2.argtypes = [c_void_p]


def cublasCreate():
    """
    Initialize CUBLAS.
    Initializes CUBLAS and creates a handle to a structure holding
    the CUBLAS library context.
    Returns
    -------
    handle : int
        CUBLAS context.
    References
    ----------
    `cublasCreate <http://docs.nvidia.com/cuda/cublas/#cublascreate>`_
    """

    handle = c_void_p()
    status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
    cublas_check_status(status)
    return handle.value


_libcublas.cublasDestroy_v2.restype = int
_libcublas.cublasDestroy_v2.argtypes = [c_void_p]


def cublasDestroy(handle):
    """
    Release CUBLAS resources.
    Releases hardware resources used by CUBLAS.
    Parameters
    ----------
    handle : int
        CUBLAS context.
    References
    ----------
    `cublasDestroy <http://docs.nvidia.com/cuda/cublas/#cublasdestroy>`_
    """

    status = _libcublas.cublasDestroy_v2(handle)
    cublas_check_status(status)


# Stream function definition
_libcublas.cublasSetStream_v2.restype = c_int
_libcublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p,
                                          ctypes.c_void_p]


def cublasSetStream(handle, id):
    """
    Set current CUBLAS library stream.
    Parameters
    ----------
    handle : id
        CUBLAS context.
    id : int
        Stream ID.
    References
    ----------
    `cublasSetStream <http://docs.nvidia.com/cuda/cublas/#cublassetstream>`_
    """

    status = _libcublas.cublasSetStream_v2(handle, id)
    cublas_check_status(status)


_libcublas.cublasGetStream_v2.restype = c_int
_libcublas.cublasGetStream_v2.argtypes = [ctypes.c_void_p,
                                          ctypes.c_void_p]


def cublasGetStream(handle):
    """
    Get current CUBLAS library stream.
    Parameters
    ----------
    handle : int
        CUBLAS context.
    Returns
    -------
    id : int
        Stream ID.
    References
    ----------
    `cublasGetStream <http://docs.nvidia.com/cuda/cublas/#cublasgetstream>`_
    """

    id = ctypes.c_void_p()
    status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(id))
    cublas_check_status(status)
    return id.value


# Set/Get matrix (async)
_libcublas.cublasSetMatrix.restype = c_int
_libcublas.cublasSetMatrix.argtypes = [
    c_int, c_int, c_int, c_void_p, c_int, c_void_p, c_int
]


def cublasSetMatrix(rows, cols, elem_size, A, lda, B, ldb):
    status = _libcublas.cublasSetMatrix(
        rows, cols, elem_size, A, int(lda), B, int(ldb))
    cublas_check_status(status)


_libcublas.cublasSetMatrixAsync.restype = c_int
_libcublas.cublasSetMatrixAsync.argtypes = [
    c_int, c_int, c_int, c_void_p, c_int, c_void_p, c_int, c_void_p
]


def cublasSetMatrixAsync(rows, cols, elem_size, A, lda, B, ldb, stream):
    status = _libcublas.cublasSetMatrixAsync(
        rows, cols, elem_size, A, int(lda), B, int(ldb), stream)
    cublas_check_status(status)


_libcublas.cublasGetMatrix.restype = c_int
_libcublas.cublasGetMatrix.argtypes = [
    c_int, c_int, c_int, c_void_p, c_int, c_void_p, c_int
]


def cublasGetMatrix(rows, cols, elem_size, A, lda, B, ldb):
    status = _libcublas.cublasGetMatrix(
        rows, cols, elem_size, A, int(lda), B, int(ldb))
    cublas_check_status(status)


_libcublas.cublasGetMatrixAsync.restype = c_int
_libcublas.cublasGetMatrixAsync.argtypes = [
    c_int, c_int, c_int, c_void_p, c_int, c_void_p, c_int, c_void_p
]


def cublasGetMatrixAsync(rows, cols, elem_size, A, lda, B, ldb, stream):
    status = _libcublas.cublasGetMatrixAsync(
        rows, cols, elem_size, A, int(lda), B, int(ldb), stream)
    cublas_check_status(status)


# DSYRK, SSYRK
_libcublas.cublasDsyrk_v2.restype = int
_libcublas.cublasDsyrk_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    status = _libcublas.cublasDsyrk_v2(handle,
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       n, k, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublas_check_status(status)


_libcublas.cublasSsyrk_v2.restype = int
_libcublas.cublasSsyrk_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    status = _libcublas.cublasSsyrk_v2(handle,
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       n, k, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(C), ldc)
    cublas_check_status(status)


# DGEMM, SGEMM
_libcublas.cublasDgemm_v2.restype = int
_libcublas.cublasDgemm_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real double precision general matrix.
    References
    ----------
    `cublas<t>gemm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm>`_
    """

    status = _libcublas.cublasDgemm_v2(handle,
                                       CUBLAS_OP[transa],
                                       CUBLAS_OP[transb], m, n, k,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublas_check_status(status)


_libcublas.cublasSgemm_v2.restype = int
_libcublas.cublasSgemm_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real single precision general matrix.
    References
    ----------
    `cublas<t>gemm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm>`_
    """

    status = _libcublas.cublasSgemm_v2(handle,
                                       CUBLAS_OP[transa],
                                       CUBLAS_OP[transb], m, n, k,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(C), ldc)
    cublas_check_status(status)


# DTRSM, STRSM
_libcublas.cublasDtrsm_v2.restype = int
_libcublas.cublasDtrsm_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real double precision triangular system with multiple right-hand sides.
    References
    ----------
    `cublas<t>trsm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trsm>`_
    """

    status = _libcublas.cublasDtrsm_v2(handle,
                                       CUBLAS_SIDE_MODE[side],
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       CUBLAS_DIAG[diag],
                                       m, n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb)
    cublas_check_status(status)


_libcublas.cublasStrsm_v2.restype = int
_libcublas.cublasStrsm_v2.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]


def cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real single precision triangular system with multiple right-hand sides.
    References
    ----------
    `cublas<t>trsm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trsm>`_
    """

    status = _libcublas.cublasStrsm_v2(handle,
                                       CUBLAS_SIDE_MODE[side],
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       CUBLAS_DIAG[diag],
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb)
    cublas_check_status(status)


_libcublas.cublasDtrsmBatched.restype = int
_libcublas.cublasDtrsmBatched.argtypes = [ctypes.c_void_p,  # handle
                                          ctypes.c_int,  # side
                                          ctypes.c_int,  # uplo
                                          ctypes.c_int,  # trans
                                          ctypes.c_int,  # diag
                                          ctypes.c_int,  # m
                                          ctypes.c_int,  # n
                                          ctypes.c_void_p,  # alpha
                                          ctypes.c_void_p,  # A
                                          ctypes.c_int,  # lda
                                          ctypes.c_void_p,  # B
                                          ctypes.c_int,  # ldb
                                          ctypes.c_int]  # batchCount


def cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount):
    status = _libcublas.cublasDtrsmBatched(handle,
                                           CUBLAS_SIDE_MODE[side],
                                           CUBLAS_FILL_MODE[uplo],
                                           CUBLAS_OP[trans],
                                           CUBLAS_DIAG[diag],
                                           m, n, int(alpha), int(A),
                                           lda, int(B), ldb, batchCount)
    cublas_check_status(status)


_libcublas.cublasStrsmBatched.restype = int
_libcublas.cublasStrsmBatched.argtypes = [ctypes.c_void_p,  # handle
                                          ctypes.c_int,  # side
                                          ctypes.c_int,  # uplo
                                          ctypes.c_int,  # trans
                                          ctypes.c_int,  # diag
                                          ctypes.c_int,  # m
                                          ctypes.c_int,  # n
                                          ctypes.c_void_p,  # alpha
                                          ctypes.c_void_p,  # A
                                          ctypes.c_int,  # lda
                                          ctypes.c_void_p,  # B
                                          ctypes.c_int,  # ldb
                                          ctypes.c_int]  # batchCount


def cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount):
    status = _libcublas.cublasStrsmBatched(handle,
                                           CUBLAS_SIDE_MODE[side],
                                           CUBLAS_FILL_MODE[uplo],
                                           CUBLAS_OP[trans],
                                           CUBLAS_DIAG[diag],
                                           m, n, int(alpha), int(A),
                                           lda, int(B), ldb, batchCount)
    cublas_check_status(status)


# DTRMM, STRMM
_libcublas.cublasStrmm_v2.restype = int
_libcublas.cublasStrmm_v2.argtypes = [ctypes.c_void_p,  # handle
                                      ctypes.c_int,  # side
                                      ctypes.c_int,  # uplo
                                      ctypes.c_int,  # trans
                                      ctypes.c_int,  # diag
                                      ctypes.c_int,  # m
                                      ctypes.c_int,  # n
                                      ctypes.c_void_p,  # alpha
                                      ctypes.c_void_p,  # A
                                      ctypes.c_int,  # lda
                                      ctypes.c_void_p,  # B
                                      ctypes.c_int,  # ldb
                                      ctypes.c_void_p,  # C
                                      ctypes.c_int]  # ldc


def cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """
    Matrix-matrix product for real single precision triangular matrix.
    References
    ----------
    `cublas<t>trmm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trmm>`_
    """
    status = _libcublas.cublasStrmm_v2(handle,
                                       CUBLAS_SIDE_MODE[side],
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       CUBLAS_DIAG[diag],
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublas_check_status(status)


_libcublas.cublasDtrmm_v2.restype = int
_libcublas.cublasDtrmm_v2.argtypes = [ctypes.c_void_p,  # handle
                                      ctypes.c_int,  # side
                                      ctypes.c_int,  # uplo
                                      ctypes.c_int,  # trans
                                      ctypes.c_int,  # diag
                                      ctypes.c_int,  # m
                                      ctypes.c_int,  # n
                                      ctypes.c_void_p,  # alpha
                                      ctypes.c_void_p,  # A
                                      ctypes.c_int,  # lda
                                      ctypes.c_void_p,  # B
                                      ctypes.c_int,  # ldb
                                      ctypes.c_void_p,  # C
                                      ctypes.c_int]  # ldc


def cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """
    Matrix-matrix product for real double precision triangular matrix.
    References
    ----------
    `cublas<t>trmm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trmm>`_
    """
    status = _libcublas.cublasDtrmm_v2(handle,
                                       CUBLAS_SIDE_MODE[side],
                                       CUBLAS_FILL_MODE[uplo],
                                       CUBLAS_OP[trans],
                                       CUBLAS_DIAG[diag],
                                       m, n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublas_check_status(status)


""" CUBLAS XT """
_libcublas.cublasXtGetBlockDim.restype = int
_libcublas.cublasXtGetBlockDim.argtypes = [ctypes.c_void_p,
                                           ctypes.c_void_p]


def cublasXtGetBlockDim(handle):
    blockDim = ctypes.c_void_p()
    status = _libcublas.cublasXtGetBlockDim(handle, ctypes.byref(blockDim))
    cublas_check_status(status)
    return blockDim.value


_libcublas.cublasXtSetBlockDim.restype = int
_libcublas.cublasXtSetBlockDim.argtypes = [ctypes.c_void_p,
                                           ctypes.c_void_p]


def cublasXtSetBlockDim(handle, blockDim):
    status = _libcublas.cublasXtSetBlockDim(handle, blockDim)
    cublas_check_status(status)


_libcublas.cublasXtStrsm.restype = int
_libcublas.cublasXtStrsm.argtypes = [ctypes.c_void_p,  # handle
                                     ctypes.c_int,  # side
                                     ctypes.c_int,  # uplo
                                     ctypes.c_int,  # trans
                                     ctypes.c_int,  # diag
                                     ctypes.c_int,  # m
                                     ctypes.c_int,  # n
                                     ctypes.c_void_p,  # alpha
                                     ctypes.c_void_p,  # A
                                     ctypes.c_int,  # lda
                                     ctypes.c_void_p,  # B
                                     ctypes.c_int,  # ldb
                                     ]


def cublasXtStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real single precision triangular system with multiple right-hand sides.
    References
    ----------
    `cublasxt<t>trsm https://docs.nvidia.com/cuda/cublas/index.html#cublasxt_trsm`_
    """

    status = _libcublas.cublasXtStrsm(handle,
                                      CUBLAS_SIDE_MODE[side],
                                      CUBLAS_FILL_MODE[uplo],
                                      CUBLAS_OP[trans],
                                      CUBLAS_DIAG[diag],
                                      m, n, ctypes.byref(ctypes.c_float(alpha)),
                                      int(A), lda, int(B), ldb)
    cublas_check_status(status)


_libcublas.cublasXtDtrsm.restype = int
_libcublas.cublasXtDtrsm.argtypes = [ctypes.c_void_p,  # handle
                                     ctypes.c_int,  # side
                                     ctypes.c_int,  # uplo
                                     ctypes.c_int,  # trans
                                     ctypes.c_int,  # diag
                                     ctypes.c_int,  # m
                                     ctypes.c_int,  # n
                                     ctypes.c_void_p,  # alpha
                                     ctypes.c_void_p,  # A
                                     ctypes.c_int,  # lda
                                     ctypes.c_void_p,  # B
                                     ctypes.c_int,  # ldb
                                     ]


def cublasXtDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real single precision triangular system with multiple right-hand sides.
    References
    ----------
    `cublasxt<t>trsm https://docs.nvidia.com/cuda/cublas/index.html#cublasxt_trsm`_
    """

    status = _libcublas.cublasXtDtrsm(handle,
                                      CUBLAS_SIDE_MODE[side],
                                      CUBLAS_FILL_MODE[uplo],
                                      CUBLAS_OP[trans],
                                      CUBLAS_DIAG[diag],
                                      m, n, ctypes.byref(ctypes.c_double(alpha)),
                                      int(A), lda, int(B), ldb)
    cublas_check_status(status)

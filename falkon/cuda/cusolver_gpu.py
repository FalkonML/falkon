import ctypes
import sys

from falkon.cuda.cublas_gpu import CUBLAS_FILL_MODE

__all__ = ["cusolverDnCreate", "cusolverDnDestroy",
           "cusolverDnSetStream", "cusolverDnGetStream",
           "cusolverDnSpotrf_bufferSize", "cusolverDnDpotrf_bufferSize",
           "cusolverDnSpotrf", "cusolverDnDpotrf"]

# Globals
CUSOLVER_EXCEPTIONS = {
    1: "CUSOLVER_STATUS_NOT_INITIALIZED",
    2: "CUSOLVER_STATUS_ALLOC_FAILED",
    3: "CUSOLVER_STATUS_INVALID_VALUE",
    4: "CUSOLVER_STATUS_ARCH_MISMATCH",
    5: "CUSOLVER_STATUS_MAPPING_ERROR",
    6: "CUSOLVER_STATUS_EXECUTION_FAILED",
    7: "CUSOLVER_STATUS_INTERNAL_ERROR",
    8: "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
    9: "CUSOLVER_STATUS_NOT_SUPPORTED",
    10: "CUSOLVER_STATUS_ZERO_PIVOT",
    11: "CUSOLVER_STATUS_INVALID_LICENSE"
}


# Library Loading
def load_cusolver_library():
    _linux_version_list = [10.2, 10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0]
    if 'linux' in sys.platform:
        _libcusolver_libname_list = ['libcusolver.so'] + \
                                    ['libcusolver.so.%s' % v for v in _linux_version_list]

        # Fix for GOMP weirdness with CUDA 8.0 on Fedora (#171):
        try:
            ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
        except:
            pass
        try:
            ctypes.CDLL('libgomp.so', mode=ctypes.RTLD_GLOBAL)
        except:
            pass
    elif sys.platform == 'darwin':
        _libcusolver_libname_list = ['libcusolver.dylib']
    else:
        raise RuntimeError('unsupported platform')

    # Print understandable error message when library cannot be found:
    _libcusolver = None
    for _libcusolver_libname in _libcusolver_libname_list:
        try:
            _libcusolver = ctypes.cdll.LoadLibrary(_libcusolver_libname)
        except OSError:
            pass
        else:
            break
    if _libcusolver == None:
        raise OSError('cusolver library not found')

    return _libcusolver


_libcusolver = load_cusolver_library()


# Error handling
class CusolverError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        try:
            self.message = CUSOLVER_EXCEPTIONS[status_code]
        except KeyError:
            self.message = "Unknown CuBLAS error %d" % (status_code)
        super().__init__(self.message)


def cusolver_check_status(status):
    """
    Raise CuSOLVER exception
    Raise an exception corresponding to the specified CuSOLVER error
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
        raise CusolverError(status)


_libcusolver.cusolverDnCreate.restype = int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.c_void_p]
def cusolverDnCreate():
    """
    Create cuSolverDn context.

    Returns
    -------
    handle : int
        cuSolverDn context.

    References
    ----------
    `cusolverDnCreate <http://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDNcreate>`_
    """

    handle = ctypes.c_void_p()
    status = _libcusolver.cusolverDnCreate(ctypes.byref(handle))
    cusolver_check_status(status)
    return handle


_libcusolver.cusolverDnDestroy.restype = int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]
def cusolverDnDestroy(handle):
    """
    Destroy cuSolverDn context.

    Parameters
    ----------
    handle : int
        cuSolverDn context.

    References
    ----------
    `cusolverDnDestroy <http://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDNdestroy>`_
    """

    status = _libcusolver.cusolverDnDestroy(handle)
    cusolver_check_status(status)


_libcusolver.cusolverDnSetStream.restype = int
_libcusolver.cusolverDnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cusolverDnSetStream(handle, stream):
    """
    Set stream used by cuSolverDN library.

    Parameters
    ----------
    handle : int
        cuSolverDN context.
    stream : int
        Stream to be used.

    References
    ----------
    `cusolverDnSetStream <http://docs.nvidia.com/cuda/cusolver/index.html#cudssetstream>`_
    """

    status = _libcusolver.cusolverDnSetStream(handle, stream)
    cusolver_check_status(status)


_libcusolver.cusolverDnGetStream.restype = int
_libcusolver.cusolverDnGetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cusolverDnGetStream(handle):
    """
    Get stream used by cuSolverDN library.

    Parameters
    ----------
    handle : int
        cuSolverDN context.

    Returns
    -------
    stream : int
        Stream used by context.

    References
    ----------
    `cusolverDnGetStream <http://docs.nvidia.com/cuda/cusolver/index.html#cudsgetstream>`_
    """

    stream = ctypes.c_int()
    status = _libcusolver.cusolverDnGetStream(handle, ctypes.byref(stream))
    cusolver_check_status(status)
    return stream.value


_libcusolver.cusolverDnSpotrf_bufferSize.restype = int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnSpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSpotrf_bufferSize(
        handle, CUBLAS_FILL_MODE[uplo], n, int(A), lda, ctypes.byref(lwork))
    cusolver_check_status(status)
    return lwork.value


_libcusolver.cusolverDnDpotrf_bufferSize.restype = int
_libcusolver.cusolverDnDpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnDpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDpotrf_bufferSize(
        handle, CUBLAS_FILL_MODE[uplo], n, int(A), lda, ctypes.byref(lwork))
    cusolver_check_status(status)
    return lwork.value


_libcusolver.cusolverDnSpotrf.restype = int
_libcusolver.cusolverDnSpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnSpotrf(handle, uplo, n, A, lda, workspace, Lwork, devInfo):
    """
    Compute Cholesky factorization of a real single precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """
    status = _libcusolver.cusolverDnSpotrf(
        handle, CUBLAS_FILL_MODE[uplo], n, int(A), lda,
        int(workspace), int(Lwork), int(devInfo.data_ptr()))

    cusolver_check_status(status)
    devInfo = devInfo.cpu().item()
    if devInfo != 0 and devInfo < n:
        raise RuntimeError("CuSolver SPOTRF error %d" % (devInfo))


_libcusolver.cusolverDnDpotrf.restype = int
_libcusolver.cusolverDnDpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnDpotrf(handle, uplo, n, A, lda, workspace, Lwork, devInfo):
    """
    Compute Cholesky factorization of a real double precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """
    status = _libcusolver.cusolverDnDpotrf(
        handle, CUBLAS_FILL_MODE[uplo], n, int(A), lda,
        int(workspace), int(Lwork), int(devInfo.data_ptr()))

    cusolver_check_status(status)
    devInfo = devInfo.cpu().item()
    if devInfo != 0 and devInfo < n:
        raise RuntimeError("CuSolver DPOTRF error %d" % (devInfo))

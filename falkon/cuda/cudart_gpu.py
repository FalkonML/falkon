import ctypes
import sys
from ctypes import c_int, c_void_p, c_size_t
from enum import Enum

import torch

__all__ = ("cuda_meminfo", "cuda_device_get_attribute", "cuda_device_can_access_peer",
           "cuda_device_enable_peer_access", "cuda_memcpy2d", "cuda_memcpy2d_async",
           "cuda_memcpy", "cuda_memcpy_async",
           "cuda_memcpy_peer", "cuda_memcpy_peer_async")


# Globals
CUDA_EXCEPTIONS = {
    1: "cudaErrorMissingConfiguration",
    2: "cudaErrorMemoryAllocation",
    3: "cudaErrorInitializationError",
    4: "cudaErrorLaunchFailure",
    5: "cudaErrorPriorLaunchFailure",
    6: "cudaErrorLaunchTimeout",
    7: "cudaErrorLaunchOutOfResources",
    8: "cudaErrorInvalidDeviceFunction",
    9: "cudaErrorInvalidConfiguration",
    10: "cudaErrorInvalidDevice",
    11: "cudaErrorInvalidValue",
    12: "cudaErrorInvalidPitchValue",
    13: "cudaErrorInvalidSymbol",
    14: "cudaErrorMapBufferObjectFailed",
    15: "cudaErrorUnmapBufferObjectFailed",
    16: "cudaErrorInvalidHostPointer",
    17: "cudaErrorInvalidDevicePointer",
    18: "cudaErrorInvalidTexture",
    19: "cudaErrorInvalidTextureBinding",
    20: "cudaErrorInvalidChannelDescriptor",
    21: "cudaErrorInvalidMemcpyDirection",
    22: "cudaError",
    23: "cudaErrorTextureFetchFailed",
    24: "cudaErrorTextureNotBound",
    25: "cudaErrorSynchronizationError",
    26: "cudaErrorInvalidFilterSetting",
    27: "cudaErrorInvalidNormSetting",
    28: "cudaErrorMixedDeviceExecution",
    29: "cudaErrorCudartUnloading",
    30: "cudaErrorUnknown",
    31: "cudaErrorNotYetImplemented",
    32: "cudaErrorMemoryValueTooLarge",
    33: "cudaErrorInvalidResourceHandle",
    34: "cudaErrorNotReady",
    35: "cudaErrorInsufficientDriver",
    36: "cudaErrorSetOnActiveProcess",
    37: "cudaErrorInvalidSurface",
    38: "cudaErrorNoDevice",
    39: "cudaErrorECCUncorrectable",
    40: "cudaErrorSharedObjectSymbolNotFound",
    41: "cudaErrorSharedObjectInitFailed",
    42: "cudaErrorUnsupportedLimit",
    43: "cudaErrorDuplicateVariableName",
    44: "cudaErrorDuplicateTextureName",
    45: "cudaErrorDuplicateSurfaceName",
    46: "cudaErrorDevicesUnavailable",
    47: "cudaErrorInvalidKernelImage",
    48: "cudaErrorNoKernelImageForDevice",
    49: "cudaErrorIncompatibleDriverContext",
    50: "cudaErrorPeerAccessAlreadyEnabled",
    51: "cudaErrorPeerAccessNotEnabled",
    52: "cudaError",
    53: "cudaError",
    54: "cudaErrorDeviceAlreadyInUse",
    55: "cudaErrorProfilerDisabled",
    56: "cudaErrorProfilerNotInitialized",
    57: "cudaErrorProfilerAlreadyStarted",
    58: "cudaErrorProfilerAlreadyStopped",
    59: "cudaErrorAssert",
    60: "cudaErrorTooManyPeers",
    61: "cudaErrorHostMemoryAlreadyRegistered",
    62: "cudaErrorHostMemoryNotRegistered",
    63: "cudaErrorOperatingSystem",
    64: "cudaErrorPeerAccessUnsupported",
    65: "cudaErrorLaunchMaxDepthExceeded",
    66: "cudaErrorLaunchFileScopedTex",
    67: "cudaErrorLaunchFileScopedSurf",
    68: "cudaErrorSyncDepthExceeded",
    69: "cudaErrorLaunchPendingCountExceeded",
    70: "cudaErrorNotPermitted",
    71: "cudaErrorNotSupported",
    72: "cudaErrorHardwareStackError",
    73: "cudaErrorIllegalInstruction",
    74: "cudaErrorMisalignedAddress",
    75: "cudaErrorInvalidAddressSpace",
    76: "cudaErrorInvalidPc",
    77: "cudaErrorIllegalAddress",
    78: "cudaErrorInvalidPtx",
    79: "cudaErrorInvalidGraphicsContext",
    127: "cudaErrorStartupFailure",
}


# Library Loading
def load_cudart_library():
    linux_version_list = [10.2, 10.1, 10.0, 9.2, 9.1, 9.0]
    if 'linux' in sys.platform:
        libcudart_libname_list = ['libcudart.so'] + \
                                 ['libcudart.so.%s' % v for v in linux_version_list]
    elif sys.platform == 'darwin':
        libcudart_libname_list = ['libcudart.dylib']
    else:
        raise RuntimeError('unsupported platform %s' % (sys.platform))

    # Print understandable error message when library cannot be found:
    libcudart = None
    for libcudart_libname in libcudart_libname_list:
        try:
            libcudart = ctypes.cdll.LoadLibrary(libcudart_libname)
        except OSError:
            pass
        else:
            break
    if libcudart is None:
        raise OSError('cublas library not found')
    return libcudart


_libcudart = load_cudart_library()


# Error Checking
_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]


def cudaGetErrorString(e):
    """
    Retrieve CUDA error string.
    Return the string associated with the specified CUDA error status
    code.
    Parameters
    ----------
    e : int
        Error number.
    Returns
    -------
    s : str
        Error string.
    """

    return _libcudart.cudaGetErrorString(e)


class CudaError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        try:
            self.message = CUDA_EXCEPTIONS[status_code]
        except KeyError:
            self.message = "Unknown CUDA error %d" % (status_code)
        super().__init__(self.message)


def cuda_check_status(status):
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
        raise CudaError(status)


# Management
def __device_index(device):
    if type(device) is torch.device:
        dd = device.index
    elif device is None:
        dd = torch.cuda.current_device()
    else:
        dd = device

    return dd


_libcudart.cudaSetDevice.restype = int
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]


def cudaSetDevice(dev):
    """
    Set current CUDA device.
    Select a device to use for subsequent CUDA operations.
    Parameters
    ----------
    dev : int
        Device number.
    """

    status = _libcudart.cudaSetDevice(dev)
    cuda_check_status(status)


_libcudart.cudaMemGetInfo.restype = int
_libcudart.cudaMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudaMemGetInfo():
    """
    Return the amount of free and total device memory.
    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.
    """

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    cuda_check_status(status)
    return free.value, total.value


def cuda_meminfo(device=None):
    dd = __device_index(device)
    cudaSetDevice(dd)
    free, total = cudaMemGetInfo()

    return free, total


# Device Attributes
class CudaDeviceAttributes(Enum):
    """Map CUDA device attributes to their codes.
    NVIDIA reference:
    https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd
    """
    cudaDevAttrKernelExecTimeout = 17
    cudaDevAttrMultiProcessorCount = 16


_libcudart.cudaDeviceGetAttribute.restype = c_int
_libcudart.cudaDeviceGetAttribute.argtypes = [c_void_p, c_int, c_int]


def cuda_device_get_attribute(device, attr):
    out_val = c_int()
    device_num = __device_index(device)
    if isinstance(attr, CudaDeviceAttributes):
        attr = attr.value

    status = _libcudart.cudaDeviceGetAttribute(ctypes.byref(out_val), attr, device_num)
    cuda_check_status(status)

    return out_val.value


def is_gpu_execution_limited(device):
    return cuda_device_get_attribute(device, CudaDeviceAttributes.cudaDevAttrKernelExecTimeout)


# Peer copies
_libcudart.cudaDeviceCanAccessPeer.restype = c_int
_libcudart.cudaDeviceCanAccessPeer.argtypes = [c_void_p, c_int, c_int]


def cuda_device_can_access_peer(device, peer_device):
    out_val = c_int()
    device = __device_index(device)
    peer_device = __device_index(peer_device)
    status = _libcudart.cudaDeviceCanAccessPeer(ctypes.byref(out_val), device, peer_device)

    cuda_check_status(status)
    return bool(out_val.value)


_libcudart.cudaDeviceEnablePeerAccess.restype = c_int
_libcudart.cudaDeviceEnablePeerAccess.argtypes = [c_int, ctypes.c_uint]


def cuda_device_enable_peer_access(device):
    device = __device_index(device)
    flags = ctypes.c_uint(0)
    status = _libcudart.cudaDeviceEnablePeerAccess(device, flags)

    cuda_check_status(status)


_libcudart.cudaMemcpyPeer.restype = c_int
_libcudart.cudaMemcpyPeer.argtypes = [c_void_p, c_int, c_void_p, c_int, c_size_t]


def cuda_memcpy_peer(dst, dst_device, src, src_device, count):
    src_device = __device_index(src_device)
    dst_device = __device_index(dst_device)

    status = _libcudart.cudaMemcpyPeer(dst, dst_device, src, src_device,
                                       c_size_t(count))
    cuda_check_status(status)


_libcudart.cudaMemcpyPeerAsync.restype = c_int
_libcudart.cudaMemcpyPeerAsync.argtypes = [c_void_p, c_int, c_void_p, c_int, c_size_t, c_void_p]


def cuda_memcpy_peer_async(dst, dst_device, src, src_device, count, stream):
    src_device = __device_index(src_device)
    dst_device = __device_index(dst_device)

    status = _libcudart.cudaMemcpyPeerAsync(
        dst, dst_device, src, src_device, c_size_t(count), stream)
    cuda_check_status(status)


_libcudart.cudaMemcpy2D.restype = c_int
_libcudart.cudaMemcpy2D.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t,
                                    c_int]


def cuda_memcpy2d(dst, dpitch, src, spitch, width, height):
    # Here 4 is cudaMemcpyDefault
    status = _libcudart.cudaMemcpy2D(dst, c_size_t(dpitch), src,
                                     c_size_t(spitch),
                                     c_size_t(width), c_size_t(height), 4)
    cuda_check_status(status)


_libcudart.cudaMemcpy2DAsync.restype = c_int
_libcudart.cudaMemcpy2DAsync.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t,
                                         c_int, c_void_p]


def cuda_memcpy2d_async(dst, dpitch, src, spitch, width, height, stream):
    # Here 4 is cudaMemcpyDefault https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
    status = _libcudart.cudaMemcpy2DAsync(dst, c_size_t(dpitch), src,
                                          c_size_t(spitch),
                                          c_size_t(width), c_size_t(height), 4,
                                          stream)
    cuda_check_status(status)


_libcudart.cudaMemcpyAsync.restype = c_int
_libcudart.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_void_p]


def cuda_memcpy_async(dst, src, count, stream):
    status = _libcudart.cudaMemcpyAsync(dst, src, c_size_t(count), 4, stream)
    cuda_check_status(status)


_libcudart.cudaMemcpy.restype = c_int
_libcudart.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]


def cuda_memcpy(dst, src, count):
    status = _libcudart.cudaMemcpy(dst, src, c_size_t(count), 4)
    cuda_check_status(status)

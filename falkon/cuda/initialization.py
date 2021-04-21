""" Handles various initialization routines for GPU """
import threading

import torch

import falkon.cuda.cusolver_gpu as cusolver
from falkon.cuda.cublas_gpu import cublasCreate, cublasDestroy
from falkon.options import BaseOptions

__all__ = ("init", "shutdown", "cublas_handle", "cusolver_handle")


def _normalize_device(device):
    # Normalize input device
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            raise RuntimeError(
                "CuBLAS handle only exists on CUDA devices. Given CPU device %s" % (device))
        device = device.index
    elif device is not None:
        device = int(device)
        if device < 0:
            raise RuntimeError(
                "Device index must be greater or equal than 0. Given index was %d" % (device))
    else:
        # Device is None
        device = torch.cuda.current_device()

    return device


_cublas_handles = {}
_cusolver_handles = {}


def cublas_handle(device=None):
    device = _normalize_device(device)
    thread = threading.current_thread().name
    name = f"{device}_{thread}"
    global _cublas_handles
    try:
        return _cublas_handles[name]
    except KeyError:
        handle = cublasCreate()
        _cublas_handles[name] = handle
        return handle


def cusolver_handle(device=None):
    device = _normalize_device(device)
    thread = threading.current_thread().name
    name = f"{device}_{thread}"
    global _cusolver_handles
    try:
        return _cusolver_handles[name]
    except KeyError:
        handle = cusolver.cusolverDnCreate()
        _cusolver_handles[name] = handle
        return handle


def init(opt: BaseOptions):
    pass


def shutdown():
    """
    Shutdown libraries used by scikit-cuda.

    Shutdown the CUBLAS, CULA, CUSOLVER, and MAGMA libraries used by
    high-level functions provided by scikits-cuda.

    Notes
    -----
    This function does not shutdown PyCUDA.
    """
    global _cublas_handles
    for i, handle in _cublas_handles.items():
        if handle is not None:
            cublasDestroy(handle)
            _cublas_handles[i] = None

    global _cusolver_handles
    for i, handle in _cusolver_handles.items():
        if handle is not None:
            cusolver.cusolverDnDestroy(handle)
            _cusolver_handles[i] = None

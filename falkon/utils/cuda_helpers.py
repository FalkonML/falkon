from typing import Optional

import torch

from .helpers import sizeof_dtype
from .tensor_helpers import is_f_contig, is_contig
from falkon.cuda.cublas_gpu import (cublasSetMatrix, cublasSetMatrixAsync,
                                    cublasGetMatrix, cublasGetMatrixAsync)


def copy_to_device(rows, cols, H, Hi, Hj, D, Di, Dj, s=None, check=False):
    if check:
        if not is_f_contig(H):
            raise ValueError("Tensor 'H' must be F-contiguous")
        if not is_f_contig(D):
            raise ValueError("Tensor 'D' must be F-contiguous")
        if rows > H.shape[0] or cols > H.shape[1]:
            raise IndexError("rows, cols (%d, %d) into H of size %s out of range." %
                             (rows, cols, H.shape))
        if rows > D.shape[0] or cols > D.shape[1]:
            raise IndexError("rows, cols (%d, %d) into D of size %s out of range." %
                             (rows, cols, D.shape))
        if H.dtype != D.dtype:
            raise ValueError("Data types of H and D (%s, %s) do not match." % (H.dtype, D.dtype))

    Hptr = H[Hi, Hj].data_ptr()
    ldh = H.stride(1)
    if H.shape[1] == 1: # col-vector
        ldh = H.shape[0] # doesn't matter, needs to be >= than rows

    Dptr = D[Di, Dj].data_ptr()
    ldd  = D.stride(1)

    dtype_size = sizeof_dtype(H.dtype)

    if s is not None:
        cublasSetMatrixAsync(
            rows=rows, cols=cols, elem_size=dtype_size,
            A=Hptr, lda=ldh, B=Dptr, ldb=ldd,
            stream=s._as_parameter_)
    else:
        cublasSetMatrix(
            rows=rows, cols=cols, elem_size=dtype_size,
            A=Hptr, lda=ldh, B=Dptr, ldb=ldd)

    return D.narrow(0, Di, rows).narrow(1, Dj, cols)


def copy_to_device_noorder(rows, cols, H, Hi, Hj, D, Di, Dj, s=None, check=False):
    if check:
        if rows > H.shape[0] or cols > H.shape[1]:
            raise IndexError("rows, cols (%d, %d) into H of size %s out of range." %
                             (rows, cols, H.shape))
        if rows > D.shape[0] or cols > D.shape[1]:
            raise IndexError("rows, cols (%d, %d) into D of size %s out of range." %
                             (rows, cols, D.shape))
        if H.dtype != D.dtype:
            raise ValueError("Data types of H and D (%s, %s) do not match." % (H.dtype, D.dtype))

    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)
    if s is not None:
        D_narrow.copy_(H_narrow, non_blocking=True)
    else:
        D_narrow.copy_(H_narrow)
    return D_narrow


def copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=None, check=False):
    if check:
        if not is_f_contig(H):
            raise ValueError("Tensor 'H' must be F-contiguous")
        if not is_f_contig(D):
            raise ValueError("Tensor 'D' must be F-contiguous")
        if rows > H.shape[0] or cols > H.shape[1]:
            raise IndexError("rows, cols (%d, %d) into H of size %s out of range." %
                             (rows, cols, H.shape))
        if rows > D.shape[0] or cols > D.shape[1]:
            raise IndexError("rows, cols (%d, %d) into D of size %s out of range." %
                             (rows, cols, D.shape))
        if H.dtype != D.dtype:
            raise ValueError("Data types of H and D (%s, %s) do not match." % (H.dtype, D.dtype))

    Hptr = H[Hi, Hj].data_ptr()
    ldh = H.stride(1)

    Dptr = D[Di, Dj].data_ptr()
    ldd  = D.stride(1)

    dtype_size = sizeof_dtype(H.dtype)

    if s is not None:
        cublasGetMatrixAsync(
            rows=rows, cols=cols, elem_size=dtype_size,
            A=Dptr, lda=ldd, B=Hptr, ldb=ldh,
            stream=s._as_parameter_)
    else:
        cublasGetMatrix(
            rows=rows, cols=cols, elem_size=dtype_size,
            A=Dptr, lda=ldd, B=Hptr, ldb=ldh)

    return H.narrow(0, Hi, rows).narrow(1, Hj, cols)


def copy_to_host_noorder(rows: int, cols: int,
                         D: torch.Tensor, Di: int, Dj: int,
                         H: torch.Tensor, Hi: int, Hj: int,
                         cpu_buf: Optional[torch.Tensor] = None,
                         s: Optional[torch.cuda.Stream] = None):
    if is_f_contig(D, strict=True):
        if cpu_buf is not None:
            if cpu_buf.shape[0] < rows or cpu_buf.shape[1] < cols:
                raise RuntimeError("Intermediate CPU Buffer is not large enough to hold data: "
                                   "expected (%d, %d); was (%d, %d)" %
                                   (rows, cols, cpu_buf.shape[0], cpu_buf.shape[1]))
            if cpu_buf.dtype != D.dtype:
                raise TypeError("Intermediate CPU buffer data type is not equal to the GPU data "
                                "type: expected %s; was %s" % (D.dtype, cpu_buf.dtype))
            restr_cpu_buf = copy_to_host(rows, cols, D, Di, Dj, cpu_buf, 0, 0, s=s, check=False)
            if s is not None:
                s.synchronize()
            H[Hi:Hi+rows, Hj:Hj+cols].copy_(restr_cpu_buf)
        else:
            copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=s)
    elif is_contig(D):
        if cpu_buf is not None:
            restr_cpu_buf = copy_to_host(cols, rows, D.T, Dj, Di, cpu_buf.T, 0, 0, s=s, check=False)
            if s is not None:
                s.synchronize()
            H[Hi:Hi+rows, Hj:Hj+cols].copy_(restr_cpu_buf.T)
        else:
            copy_to_host(cols, rows, D.T, Dj, Di, H.T, Hj, Hi, s=s)
    else:
        raise RuntimeError("Cannot copy data which is not memory contiguous.")

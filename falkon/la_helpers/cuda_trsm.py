from typing import Optional

import torch

from falkon.cuda.initialization import cublas_handle
from falkon.cuda.cublas_gpu import cublasStrsm, cublasDtrsm, cublas_stream
from falkon.utils.helpers import choose_fn, check_same_device
from falkon.c_ext import copy_transpose
from falkon.utils.tensor_helpers import is_f_contig, create_fortran, create_C


def cuda_trsm(A: torch.Tensor, v: torch.Tensor, alpha: float, lower: int, transpose: int,
              stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
    if not is_f_contig(A, strict=False):
        raise ValueError("A must be f-contiguous for CUDA TRSM to work.")
    if not check_same_device(A, v):
        raise ValueError("A and v must be on the same CUDA device.")
    if not A.is_cuda:
        raise ValueError("A and v must be CUDA tensors!")

    device = A.device
    s = stream
    if stream is None:
        s = torch.cuda.current_stream(device=device)
    cublas_hdl = cublas_handle(device.index)
    trsm_fn = choose_fn(A.dtype, cublasDtrsm, cublasStrsm, "TRSM")

    # noinspection PyProtectedMember
    with torch.cuda.device(device), torch.cuda.stream(s), cublas_stream(cublas_hdl, s._as_parameter_):
        # Deal with copying v, which may not be F-contiguous.
        vF = create_fortran(v.size(), v.dtype, device)
        if is_f_contig(v, strict=False):
            # We can just make a copy of v
            vF.copy_(v)
            s.synchronize()  # sync is necessary here for correctness. Not sure why! TODO: Is it still needed?
        else:
            vF = copy_transpose(input=v, output=vF.T).T

        uplo = 'L' if lower else 'U'
        trans = 'T' if transpose else 'N'
        trsm_fn(cublas_hdl, side='L', uplo=uplo, trans=trans, diag='N', m=vF.shape[0], n=vF.shape[1],
                alpha=alpha, A=A.data_ptr(), lda=A.stride(1), B=vF.data_ptr(), ldb=vF.stride(1))
        if is_f_contig(v, strict=False):
            vout = vF
        else:
            vout = create_C(v.size(), v.dtype, device)
            vout = copy_transpose(input=vF, output=vout.T).T
    return vout

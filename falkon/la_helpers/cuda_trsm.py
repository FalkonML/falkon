from typing import Optional

import torch

from falkon.c_ext import copy_transpose, cublas_trsm
from falkon.utils.helpers import check_same_device
from falkon.utils.tensor_helpers import create_C, create_fortran, is_f_contig


def cuda_trsm(
    A: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    lower: bool,
    transpose: bool,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
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

    # noinspection PyProtectedMember
    with torch.cuda.device(device), torch.cuda.stream(s):
        # Deal with copying v, which may not be F-contiguous.
        vF = create_fortran(v.size(), v.dtype, device)
        if is_f_contig(v, strict=False):
            # We can just make a copy of v
            vF.copy_(v)
            s.synchronize()  # sync is necessary here for correctness. Not sure why! TODO: Is it still needed?
        else:
            vF = copy_transpose(v, out=vF.T).T

        cublas_trsm(
            A=A,
            lda=A.stride(1),
            B=vF,
            ldb=vF.stride(1),
            alpha=alpha,
            left=True,
            upper=not lower,
            transpose=transpose,
            unitriangular=False,
            m=vF.shape[0],
            n=vF.shape[1],
        )
        if is_f_contig(v, strict=False):
            vout = vF
        else:
            vout = create_C(v.size(), v.dtype, device)
            vout = copy_transpose(vF, out=vout.T).T
    return vout

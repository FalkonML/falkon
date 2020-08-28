import torch

from falkon.cuda.initialization import cublas_handle
from falkon.cuda.cublas_gpu import cublasStrsm, cublasDtrsm, cublasSetStream
from falkon.utils.helpers import choose_fn, check_same_device
# noinspection PyUnresolvedReferences
from falkon.la_helpers.cuda_la_helpers import cuda_transpose
from falkon.utils.tensor_helpers import is_f_contig, create_fortran, create_C


def cuda_trsm(A: torch.Tensor, v: torch.Tensor, alpha: float, lower: int, transpose: int) -> torch.Tensor:
    if not is_f_contig(A, strict=False):
        raise ValueError("A must be f-contiguous for CUDA TRSM to work.")
    if not check_same_device(A, v):
        raise ValueError("A and v must be on the same CUDA device.")
    if not A.is_cuda:
        raise ValueError("A and v must be CUDA tensors!")

    s1 = torch.cuda.Stream(device=A.device)
    cublas_hdl = cublas_handle(A.device.index)
    cublasSetStream(cublas_hdl, s1._as_parameter_)

    with torch.cuda.device(A.device), torch.cuda.stream(s1):
        # Deal with copying v, which may not be F-contiguous.
        vF = create_fortran(v.size(), v.dtype, v.device)
        if is_f_contig(v, strict=False):
            # We can just make a copy of v
            vF.copy_(v)
        else:
            vF = cuda_transpose(input=v, output=vF.T).T

        trsm_fn = choose_fn(A.dtype, cublasDtrsm, cublasStrsm, "TRSM")
        handle = cublas_handle(A.device)

        uplo = 'L' if lower else 'U'
        trans = 'T' if transpose else 'N'
        trsm_fn(handle, side='L', uplo=uplo, trans=trans, diag='N', m=vF.shape[0], n=vF.shape[1],
                alpha=alpha, A=A.data_ptr(), lda=A.stride(1), B=vF.data_ptr(), ldb=vF.stride(1))
        if not is_f_contig(v, strict=False):
            vout = create_C(v.size(), v.dtype, v.device)
            vout = cuda_transpose(input=vF, output=vout.T).T
            s1.synchronize()
            return vout
        s1.synchronize()
        return vF

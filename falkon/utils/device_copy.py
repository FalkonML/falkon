import torch.cuda

from .helpers import sizeof_dtype
from .tensor_helpers import is_f_contig, is_contig, is_contig_vec

if torch.cuda.is_available():
    # from falkon.cuda.cudart_gpu import (
    #     cuda_memcpy2d, cuda_memcpy2d_async,
    #     cuda_memcpy, cuda_memcpy_async
    # )
    # from falkon.cuda.cublas_gpu import (cublasSetMatrix, cublasSetMatrixAsync,
    #                                     cublasGetMatrix, cublasGetMatrixAsync)
    from falkon.c_ext import cublas_2d_copy_to_dev_async as cublasSetMatrixAsync
    from falkon.c_ext import cublas_2d_copy_to_dev as cublasSetMatrix
    from falkon.c_ext import cublas_2d_copy_to_host_async as cublasGetMatrixAsync
    from falkon.c_ext import cublas_2d_copy_to_host as cublasGetMatrix
    from falkon.c_ext import cuda_2d_copy_async, cuda_2d_copy, cuda_1d_copy_async, cuda_1d_copy


def check_copy(origin, dest, check_dtypes=True):
    if check_dtypes:
        # Data-types
        if origin.dtype != dest.dtype:
            raise ValueError("Data types of origin and destination (%s, %s) do not match." % (
                origin.dtype, dest.dtype))
    # Sizes
    if origin.size() != dest.size():
        raise ValueError("Size of origin (%s) does not match size of destination (%s)" % (
            origin.size(), dest.size()))
    # Contiguity
    if is_f_contig(origin, strict=False):
        if not is_f_contig(dest, strict=False):
            raise ValueError(
                "origin is F-contig (strides %s), while destination is not (strides %s)" % (
                    origin.stride(), dest.stride()))
    elif is_contig(origin):  # H is C-contiguous
        if not is_contig(dest) or is_f_contig(dest, strict=True):
            raise ValueError(
                "origin is C-contig (strides %s), while destination is not (strides %s)" % (
                    origin.stride(), dest.stride()))
    else:
        raise ValueError("origin is not memory-contiguous (strides %s)" % (origin.stride(),))


def copy(origin, dest, s=None, allow_dtype_change=False):
    check_copy(origin, dest, check_dtypes=not allow_dtype_change)

    if origin.device.type == dest.device.type:
        dest.copy_(origin)
    elif origin.device.type == "cpu":  # host -> dev
        copy_to_device(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    else:  # dev -> host
        copy_to_host(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    return dest


# noinspection PyProtectedMember
def copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=None):
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)

    H_narrow_final = None
    if H.dtype != D.dtype:
        # First copy `D_narrow` to a matrix with the same dtype on the host
        # then copy this last matrix to `H_narrow` (at the end of this function).
        H_temp = torch.empty_like(D_narrow, device=H.device)
        H_narrow_final = H_narrow
        H_narrow = H_temp

    dts = sizeof_dtype(D.dtype)

    if is_contig_vec(H_narrow) and is_contig_vec(D_narrow):
        if s is not None:
            # cuda_memcpy_async(
            #     src=D_narrow.data_ptr(), dst=H_narrow.data_ptr(),
            #     count=(rows * cols) * dts, stream=s._as_parameter_)
            cuda_1d_copy_async(
                src_tensor=D_narrow, dest_tensor=H_narrow, count=(rows * cols) * dts, stream=s._as_parameter_)
        else:
            # cuda_memcpy(
            #     src=D_narrow.data_ptr(), dst=H_narrow.data_ptr(), count=(rows * cols) * dts)
            cuda_1d_copy(
                src_tensor=D_narrow, dest_tensor=H_narrow, count=(rows * cols) * dts)
    elif is_f_contig(D, strict=True):
        if s is not None:
            # cublasGetMatrixAsync(
            #     rows=rows, cols=cols, elem_size=dts,
            #     A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
            #     B=H_narrow.data_ptr(), ldb=H_narrow.stride(1),
            #     stream=s._as_parameter_)
            cublasGetMatrixAsync(rows, cols, dts, D_narrow, D_narrow.stride(1), H_narrow,
                                 H_narrow.stride(1), s._as_parameter_)
        else:
            # cublasGetMatrix(
            #     rows=rows, cols=cols, elem_size=dts,
            #     A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
            #     B=H_narrow.data_ptr(), ldb=H_narrow.stride(1))
            cublasGetMatrix(rows, cols, dts, D_narrow, D_narrow.stride(1), H_narrow,
                            H_narrow.stride(1))
    elif is_contig(D):
        if s is not None:
            cuda_2d_copy_async(
                src_tensor=D_narrow, src_pitch=D_narrow.stride(0) * dts,
                dest_tensor=H_narrow, dest_pitch=H_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter_)
            # cuda_memcpy2d_async(
            #     dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
            #     src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
            #     width=cols * dts, height=rows, stream=s._as_parameter_)
        else:
            cuda_2d_copy(
                src_tensor=D_narrow, src_pitch=D_narrow.stride(0) * dts,
                dest_tensor=H_narrow, dest_pitch=H_narrow.stride(0) * dts,
                width=cols * dts, height=rows)
            # cuda_memcpy2d(
            #     dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
            #     src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
            #     width=cols * dts, height=rows)

    if H.dtype != D.dtype:
        H_narrow_final.copy_(H_narrow)  # Blocking copy since it's H->H.


# noinspection PyProtectedMember
def copy_to_device(rows, cols, H, Hi, Hj, D, Di, Dj, s=None):
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)

    if H.dtype != D.dtype:
        # First copy `H_narrow` to another matrix with correct dtype also on host
        # then copy this last matrix to `D`.
        H_right_dt = torch.empty_like(H_narrow, dtype=D.dtype)
        H_right_dt.copy_(H)  # Copy here will be blocking since it's H->H
        H_narrow = H_right_dt

    dts = sizeof_dtype(D.dtype)
    if is_contig_vec(H_narrow) and is_contig_vec(D_narrow):
        if s is not None:
            cuda_1d_copy_async(
                src_tensor=H_narrow, dest_tensor=D_narrow, count=(rows * cols) * dts, stream=s._as_parameter_)
            # cuda_memcpy_async(
            #     src=H_narrow.data_ptr(), dst=D_narrow.data_ptr(),
            #     count=(rows * cols) * dts, stream=s._as_parameter_)
        else:
            # cuda_memcpy(
            #     src=H_narrow.data_ptr(), dst=D_narrow.data_ptr(), count=(rows * cols) * dts)
            cuda_1d_copy(
                src_tensor=H_narrow, dest_tensor=D_narrow, count=(rows * cols) * dts)
    elif is_f_contig(H, strict=True):
        if s is not None:
            # cublasSetMatrixAsync(
            #     rows=rows, cols=cols, elem_size=dts,
            #     A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
            #     B=D_narrow.data_ptr(), ldb=D_narrow.stride(1),
            #     stream=s._as_parameter_)
            cublasSetMatrixAsync(rows, cols, dts, H_narrow, H_narrow.stride(1), D_narrow,
                                 D_narrow.stride(1), s)
        else:
            # cublasSetMatrix(
            #     rows=rows, cols=cols, elem_size=dts,
            #     A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
            #     B=D_narrow.data_ptr(), ldb=D_narrow.stride(1))
            cublasSetMatrix(rows, cols, dts, H_narrow, H_narrow.stride(1), D_narrow,
                            D_narrow.stride(1))
    elif is_contig(H):
        if s is not None:
            # cuda_memcpy2d_async(
            #     src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
            #     dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
            #     width=cols * dts, height=rows, stream=s._as_parameter_)
            cuda_2d_copy_async(
                src_tensor=H_narrow, src_pitch=H_narrow.stride(0) * dts,
                dest_tensor=D_narrow, dest_pitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter_)
        else:
            # cuda_memcpy2d(
            #     src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
            #     dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
            #     width=cols * dts, height=rows)
            cuda_2d_copy(
                src_tensor=H_narrow, src_pitch=H_narrow.stride(0) * dts,
                dest_tensor=D_narrow, dest_pitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows)
    return D_narrow

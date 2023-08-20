"""
Taken from nerfacc (https://github.com/KAIR-BAIR/nerfacc) (MIT Licence)

Copyright (c) 2022 Ruilong Li, UC Berkeley.
Copyright (c) 2023 Giacomo Meanti
"""

from typing import Callable

import torch


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _assert_has_ext

        _assert_has_ext()
        return getattr(torch.ops.falkon, name)(*args, **kwargs)

    return call_cuda


# Custom la functions
parallel_potrf = _make_lazy_cuda_func("parallel_potrf")
lauum_cuda = _make_lazy_cuda_func("lauum")

# Triangular helpers
copy_triang = _make_lazy_cuda_func("copy_triang")
mul_triang = _make_lazy_cuda_func("mul_triang")
copy_transpose = _make_lazy_cuda_func("copy_transpose")
vec_mul_triang = _make_lazy_cuda_func("vec_mul_triang")

# Sparse matrices
spspmm = _make_lazy_cuda_func("spspmm")
csr2dense = _make_lazy_cuda_func("csr2dense")
sparse_row_norm_sq = _make_lazy_cuda_func("sparse_square_norm")
sparse_row_norm = _make_lazy_cuda_func("sparse_norm")
sparse_bdot = _make_lazy_cuda_func("sparse_bdot")

# Square norm with autograd
square_norm = _make_lazy_cuda_func("square_norm")

# Wrappers
cublas_2d_copy_to_dev_async = _make_lazy_cuda_func("cublas_2d_copy_to_dev_async")
cublas_2d_copy_to_dev = _make_lazy_cuda_func("cublas_2d_copy_to_dev")
cublas_2d_copy_to_host_async = _make_lazy_cuda_func("cublas_2d_copy_to_host_async")
cublas_2d_copy_to_host = _make_lazy_cuda_func("cublas_2d_copy_to_host")
cuda_2d_copy_async = _make_lazy_cuda_func("cuda_2d_copy_async")
cuda_2d_copy = _make_lazy_cuda_func("cuda_2d_copy")
cuda_1d_copy_async = _make_lazy_cuda_func("cuda_1d_copy_async")
cuda_1d_copy = _make_lazy_cuda_func("cuda_1d_copy")
mem_get_info = _make_lazy_cuda_func("mem_get_info")
cusolver_potrf_buffer_size = _make_lazy_cuda_func("cusolver_potrf_buffer_size")
cusolver_potrf = _make_lazy_cuda_func("cusolver_potrf")
potrf = _make_lazy_cuda_func("potrf")
cublas_trsm = _make_lazy_cuda_func("cublas_trsm")
cublas_trmm = _make_lazy_cuda_func("cublas_trmm")
cublas_gemm = _make_lazy_cuda_func("cublas_gemm")
cublas_syrk = _make_lazy_cuda_func("cublas_syrk")
cuda_version = _make_lazy_cuda_func("_cuda_version")

from typing import Optional

import torch
from torch import Tensor

from falkon import FalkonOptions
from falkon.kernels import Kernel
from falkon.mmv_ops.fmmv_incore import incore_fdmmv, incore_fmmv
from falkon.sparse import SparseTensor
from falkon.utils.helpers import check_same_dtype


class PrecomputedKernel(Kernel):
    def __init__(self, k: Tensor, opt: Optional[FalkonOptions] = None):
        super().__init__("precomputed", opt)
        self.k = k

    def compute(self, X1: Tensor, X2: Tensor, out: Tensor, diag: bool, **kwargs) -> Tensor:
        raise NotImplementedError()

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: Tensor, diag: bool, **kwargs) -> Tensor:
        raise NotImplementedError()

    def _decide_mmv_impl(self, X1, X2, v, opt):
        return self.mmv_impl

    def mmv_impl(self, X1, X2, v, out, opt, **kwargs) -> Tensor:
        # decide whether we must transpose based on shapes of X1, X2. No error checking here
        if self.k.shape[1] == v.shape[0]:
            transpose = False
        else:
            transpose = True
        return incore_fmmv(self.k, v, out, transpose=transpose, opt=opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        return self.dmmv_impl

    def dmmv_impl(self, v, w, out, opt, **kwargs) -> Tensor:
        return incore_fdmmv(self.k, v, w, out, opt=opt)

    def _decide_mm_impl(self, X1, X2, diag, opt):
        return self.mm_impl

    def mm_impl(self, out: Optional[Tensor], diag: bool, **kwargs) -> Tensor:
        k = self.k
        if diag:
            k = torch.diagonal(k)
        if out is not None:
            return out.copy_(k)
        return k

    @staticmethod
    def _check_device_properties(*args, fn_name: str, opt: FalkonOptions):
        pass

    @staticmethod
    def _check_mm_dimensions(X1: torch.Tensor, X2: torch.Tensor, diag: bool, out: Optional[torch.Tensor]):
        return X1, X2, out

    @staticmethod
    def _check_mmv_dimensions(X1: torch.Tensor, X2: torch.Tensor, v: torch.Tensor, out: Optional[torch.Tensor]):
        if v.dim() == 1:
            v = v.reshape((-1, 1))
        if v.dim() != 2:
            raise ValueError(f"v must be a vector or a 2D matrix. Found {len(v.shape)}D.")

        if not check_same_dtype(v, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, v, out

    @staticmethod
    def _check_dmmv_dimensions(X1, X2, v, w, out):
        # Parameter validation
        if v is None and w is None:
            raise ValueError("One of v and w must be specified to run fdMMV.")

        if v is not None and v.dim() == 1:
            v = v.reshape((-1, 1))
        if v is not None and v.dim() != 2:
            raise ValueError(f"v must be a vector or a 2D matrix. Found {len(v.shape)}D.")
        if w is not None and w.dim() == 1:
            w = w.reshape((-1, 1))
        if w is not None and w.dim() != 2:
            raise ValueError(f"w must be a vector or a 2D matrix. Found {len(w.shape)}D.")

        if not check_same_dtype(v, w, out):
            raise TypeError("Data types of input matrices must be equal.")

        return X1, X2, v, w, out

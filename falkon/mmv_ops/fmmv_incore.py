from typing import Optional

import torch

from falkon.options import FalkonOptions
from falkon.utils.helpers import check_same_device, check_same_dtype
from falkon.utils.tensor_helpers import create_same_stride

__all__ = (
    "incore_fmmv",
    "incore_fdmmv",
)


def incore_fmmv(
    mat: torch.Tensor,
    vec: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transpose: bool = False,
    opt: Optional[FalkonOptions] = None,
) -> torch.Tensor:
    if not check_same_dtype(mat, vec, out):
        raise TypeError("Data types of input matrices must be equal.")
    if not check_same_device(mat, vec, out):
        raise RuntimeError("All input arguments to incore_fmmv must be on the same device")

    if out is None:
        if transpose:
            out_shape = (mat.shape[1], vec.shape[1])
        else:
            out_shape = (mat.shape[0], vec.shape[1])
        out = create_same_stride(out_shape, mat, mat.dtype, device=mat.device, pin_memory=False)
    out.fill_(0.0)

    if transpose:
        out.addmm_(mat.T, vec, beta=0.0)
    else:
        out.addmm_(mat, vec, beta=0.0)
    return out


def incore_fdmmv(
    mat: torch.Tensor,
    vec: torch.Tensor,
    w: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    opt: Optional[FalkonOptions] = None,
) -> torch.Tensor:
    out1 = incore_fmmv(mat, vec, None, False, opt)
    if w is not None:
        out1.add_(w)
    return incore_fmmv(mat, out1, out, True, opt)

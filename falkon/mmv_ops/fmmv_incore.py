
from typing import Optional

import torch

from falkon.options import FalkonOptions
from falkon.utils.helpers import check_same_dtype, check_same_device
from falkon.utils.tensor_helpers import create_same_stride

__all__ = ("incore_fmmv", "incore_fdmmv", )



def incore_fmmv(mat: torch.Tensor, vec: torch.Tensor, out: Optional[torch.Tensor] = None, opt: Optional[FalkonOptions] = None) -> torch.Tensor:
    if not check_same_dtype(mat, vec, out):
        raise TypeError("Data types of input matrices must be equal.")
    if not check_same_device(mat, vec, out):
        raise RuntimeError("All input arguments to incore_fmmv must be on the same device")

    if out is None:
        out = create_same_stride((mat.shape[0], vec.shape[1]), mat, mat.dtype, device=mat.device,
                                 pin_memory=False)


    if mat.is_cuda:
        s1 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            out.addmm_(mat, vec, beta=0.0)
            s1.synchronize()
    else:
        out.addmm_(mat, vec, beta=0.0)
    return out


def incore_fdmmv(mat: torch.Tensor, vec: torch.Tensor, out: Optional[torch.Tensor] = None, opt: Optional[FalkonOptions] = None) -> torch.Tensor:
    out1 = incore_fmmv(mat, vec, None, opt)
    return incore_fmmv(mat.T, out1, out, opt)


from copy import deepcopy
from typing import Union
import torch

from falkon.la_helpers.square_norm_fn import square_norm_diff


@torch.jit.script
def squared_euclidean_distance(x1, x2):
    x1_norm = torch.norm(x1, p=2, dim=-1, keepdim=True).pow(2)
    x2_norm = torch.norm(x2, p=2, dim=-1, keepdim=True).pow(2)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30)
    return res


#@torch.jit.script
def full_rbf_kernel(X1, X2, sigma):
    mat1_div_sig = X1 / sigma  # n*d
    mat2_div_sig = X2 / sigma  # m*d
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, dim=-1, keepdim=True)  # n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, dim=-1, keepdim=True)  # m*1

    out = torch.addmm(norm_sq_mat1, mat1_div_sig, mat2_div_sig.T, alpha=-2, beta=1)  # n*m
    out.add_(norm_sq_mat2.T)
    out.clamp_min_(1e-30)
    out.mul_(-0.5)
    out.exp_()
    return out


def get_scalar(t: Union[torch.Tensor, float]) -> float:
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            return deepcopy(t.detach().cpu().item())
        return deepcopy(torch.flatten(t)[0].detach().cpu().item())
    return t

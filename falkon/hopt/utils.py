from copy import deepcopy
from typing import Union
import torch


def get_scalar(t: Union[torch.Tensor, float]) -> float:
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            return deepcopy(t.detach().cpu().item())
        return deepcopy(torch.flatten(t)[0].detach().cpu().item())
    return t

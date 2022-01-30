import abc
import functools
from typing import Dict, Any

import torch
from torch import nn

from falkon.kernels import KeopsKernelMixin, Kernel
from falkon.options import FalkonOptions


class DiffKernel(Kernel, KeopsKernelMixin, abc.ABC):
    def __init__(self, name, options, core_fn, **kernel_params):
        super(DiffKernel, self).__init__(name=name, kernel_type="distance", opt=options)
        self.core_fn = core_fn
        self._other_params = {}
        for k, v in kernel_params.items():
            if isinstance(v, torch.Tensor):
                self.register_parameter(k, nn.Parameter(v, requires_grad=v.requires_grad))
                # self.register_buffer(k, v)
            else:
                self._other_params[k] = v
                setattr(self, k, v)
        # self._tensor_params = {k: v for k, v in kernel_params.items() if isinstance(v, torch.Tensor)}
        # self._other_params = {k: v for k, v in kernel_params.items() if not isinstance(v, torch.Tensor)}

    @property
    def diff_params(self) -> Dict[str, torch.Tensor]:
        # return dict(self.named_buffers())
        return dict(self.named_parameters())

    @property
    def nondiff_params(self) -> Dict[str, Any]:
        return self._other_params

    @abc.abstractmethod
    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        pass

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        return self.core_fn(X1, X2, out, **self.diff_params, **self._other_params)

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor):
        return self.core_fn(X1, X2, out=None, **self.diff_params, **self._other_params)

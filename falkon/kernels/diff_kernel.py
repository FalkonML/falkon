import abc
import dataclasses
from typing import Dict, Any, Union, Optional

import torch
from torch import nn

import falkon
from falkon.kernels import Kernel
from falkon.sparse import SparseTensor
from falkon.utils.helpers import check_sparse


class DiffKernel(Kernel, abc.ABC):
    """Abstract class for differentiable kernels.

    This class should be extended instead of :class:`~falkon.kernels.kernel.Kernel` whenever designing
    a custom kernel to be used with automatic hyperparameter optimization (see the :mod:`~falkon.hopt`
    module).

    Subclasses should implement the :meth:`detach` method to return a new instance of the kernel,
    with its parameters detached from the computation graph.

    The :meth:`compute_diff` method should be overridden, unless the ``core_fn`` parameter is
    passed to the constructor.

    Hyperparameters to the concrete kernel instance (for example the length-scale of the Gaussian
    kernel) should be passed to the constructor of this class, in order to be registered
    as parameters of the computation graph. Even non-differentiable parameters should be provided
    as keywords (also non tensor arguments).

    Parameters
    ----------
    name
        A short name for the kernel (e.g. "Gaussian")
    options
        Base set of options to be used for operations involving this kernel.
    core_fn
        Optional function which can be used to compute a kernel matrix.
        The signature of the function should be:
        ``core_fn(X1, X2, out, diag, **kernel_parameters)```
        where ``X1`` and ``X2`` are the input matrices, ``out`` is the output matrix (it will
        be ``None`` when called from :meth:`compute_diff`), ``diag`` is a flag indicating
        that only the diagonal of the kernel matrix is to be computed, and ``**kernel_parameters``
        includes all additional parameters belonging to the kernel (which are passed to the
        constructor of :class:`DiffKernel`).
    kernel_params
        All parameters (differentiable and non-differentiable) to be used for this kernel.
        The values given are used to initialize the actual parameters - which will be copied in
        the constructor.
    """
    def __init__(self, name, options, core_fn, **kernel_params):
        super().__init__(name=name, opt=options)
        self.core_fn = core_fn
        self._other_params = {}
        for k, v in kernel_params.items():
            if isinstance(v, torch.Tensor):
                self.register_parameter(k, nn.Parameter(v, requires_grad=v.requires_grad))
            else:
                self._other_params[k] = v
                setattr(self, k, v)

    @property
    def diff_params(self) -> Dict[str, torch.Tensor]:
        """
        A dictionary mapping parameter names to their values for all **differentiable** parameters
        of the kernel.

        Returns
        -------
        params :
            A dictionary mapping parameter names to their values
        """
        return dict(self.named_parameters())

    @property
    def nondiff_params(self) -> Dict[str, Any]:
        """
        A dictionary mapping parameter names to their values for all **non-differentiable**
        parameters of the kernel.

        Returns
        -------
        params :
            A dictionary mapping parameter names to their values
        """
        return self._other_params

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):
        return self.core_fn(X1, X2, out, **self.diff_params, diag=diag, **self._other_params)

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        """
        Compute the kernel matrix of ``X1`` and ``X2``. The output should be differentiable with
        respect to `X1`, `X2`, and all kernel parameters returned by the :meth:`diff_params` method.

        Parameters
        ----------
        X1 : torch.Tensor
            The left matrix for computing the kernel
        X2 : torch.Tensor
            The right matrix for computing the kernel
        diag : bool
            If true, ``X1`` and ``X2`` have the same shape, and only the diagonal of ``k(X1, X2)``
            is to be computed and stored in ``out``. Otherwise compute the full kernel matrix.

        Returns
        -------
        out : torch.Tensor
            The constructed kernel matrix.
        """
        return self.core_fn(X1, X2, out=None, diag=diag, **self.diff_params, **self._other_params)

    @abc.abstractmethod
    def detach(self) -> 'Kernel':
        """Detaches all differentiable parameters of the kernel from the computation graph.

        Returns
        -------
        k :
            A new instance of the kernel sharing the same parameters, but detached from the
            computation graph.
        """
        pass

    def dmmv(self,
             X1: Union[torch.Tensor, SparseTensor],
             X2: Union[torch.Tensor, SparseTensor],
             v: Optional[torch.Tensor],
             w: Optional[torch.Tensor], out: Optional[torch.Tensor] = None,
             opt: Optional['falkon.FalkonOptions'] = None):
        X1, X2, v, w, out = self._check_dmmv_dimensions(X1, X2, v, w, out)
        params = self.params
        if opt is not None:
            params = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        dmmv_impl = self._decide_dmmv_impl(X1, X2, v, w, params)
        sparsity = check_sparse(X1, X2)
        diff = (
            (not any(sparsity)) and
            any(
                t.requires_grad for t in [X1, X2, v, w] + list(self.diff_params.values())
                if t is not None
            )
        )
        return dmmv_impl(X1, X2, v, w, self, out, diff, params)

import warnings
from typing import Union, Optional, Dict

import torch

from falkon import sparse
from falkon.kernels.diff_kernel import DiffKernel
from falkon.options import FalkonOptions
from falkon.sparse import SparseTensor


def validate_diff_float(num: Union[float, torch.Tensor], param_name: str) -> torch.Tensor:
    if isinstance(num, torch.Tensor):
        # Sigma is a 1-item tensor ('single')
        try:
            num.item()
            return num
        except ValueError:
            raise ValueError(f"Parameter {param_name} must be a scalar.")
    else:
        try:
            return torch.tensor([float(num)], dtype=torch.float64)
        except TypeError:
            raise TypeError(f"Parameter {param_name} must be a scalar or a tensor.")


def linear_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    return out


def linear_core_sparse(mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor,
                       beta, gamma) -> torch.Tensor:
    sparse.sparse_matmul(mat1, mat2, out)
    out.mul_(gamma)
    out.add_(beta)
    return out


def polynomial_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma, degree):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    out.pow_(degree)
    return out


def polynomial_core_sparse(mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor,
                           beta, gamma, degree) -> torch.Tensor:
    sparse.sparse_matmul(mat1, mat2, out)
    out.mul_(gamma)
    out.add_(beta)
    out.pow_(degree)
    return out


def sigmoid_core(mat1, mat2, out: Optional[torch.Tensor], beta, gamma):
    if out is None:
        out = torch.mm(mat1, mat2.T)
    else:
        out = torch.mm(mat1, mat2.T, out=out)
    out.mul_(gamma)
    out.add_(beta)
    out.tanh_()
    return out


def sigmoid_core_sparse(mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor,
                        beta, gamma) -> torch.Tensor:
    sparse.sparse_matmul(mat1, mat2, out)
    out.mul_(gamma)
    out.add_(beta)
    out.tanh_()
    return out


class LinearKernel(DiffKernel):
    """Linear Kernel with optional scaling and translation parameters.

    The kernel implemented here is the covariance function in the original
    input space (i.e. `X @ X.T`) with optional parameters to translate
    and scale the kernel: `beta + 1/(sigma**2) * X @ X.T`

    Parameters
    -----------
    beta : float-like
        Additive constant for the kernel, default: 0.0
    gamma : float-like
        Multiplicative constant for the kernel. The kernel will
        be multiplied by the inverse of sigma squared. Default: 1.0
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.

    Examples
    --------
    >>> k = LinearKernel(beta=0.0, sigma=2.0)
    >>> X = torch.randn(100, 3)  # 100 samples in 3 dimensions
    >>> kernel_matrix = k(X, X)
    >>> torch.testing.assert_allclose(kernel_matrix, X @ X.T * (1/2**2))
    """

    def __init__(self,
                 beta: Union[float, torch.Tensor] = 0.0,
                 gamma: Union[float, torch.Tensor] = 1.0,
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_float(beta, param_name="beta")
        self.gamma = validate_diff_float(gamma, param_name="gamma")
        super().__init__("Linear", opt, linear_core, beta=self.beta, gamma=self.gamma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = '(gamma * (X | Y) + beta) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'gamma = Pm(1)',
            'beta = Pm(1)',
        ]
        other_vars = [
            self.gamma.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.beta.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
        ]
        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'LinearKernel':
        detached_params = self._detach_params()
        return LinearKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                            opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        return linear_core_sparse(X1, X2, out,
                                  beta=dev_kernel_tensor_params["beta"],
                                  gamma=dev_kernel_tensor_params["gamma"])

    def __str__(self):
        return f"LinearKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()


class PolynomialKernel(DiffKernel):
    r"""Polynomial kernel with multiplicative and additive constants.

    Follows the formula

    .. math::

        (\alpha * X_1^\top X_2 + \beta)^{\mathrm{degree}}

    Where all operations apart from the matrix multiplication are taken element-wise.

    Parameters
    ----------
    beta : float-like
        Additive constant
    gamma : float-like
        Multiplicative constant
    degree : float-like
        Power of the polynomial kernel
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.
    """

    def __init__(self,
                 beta: Union[float, torch.Tensor],
                 gamma: Union[float, torch.Tensor],
                 degree: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_float(beta, param_name="beta")
        self.gamma = validate_diff_float(gamma, param_name="gamma")
        self.degree = validate_diff_float(degree, param_name="degree")
        super().__init__("Polynomial", opt, polynomial_core, beta=self.beta, gamma=self.gamma,
                         degree=self.degree)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        formula = 'Powf((gamma * (X | Y) + beta), degree) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'gamma = Pm(1)',
            'beta = Pm(1)',
            'degree = Pm(1)',
        ]
        other_vars = [
            self.gamma.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.beta.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.degree.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
        ]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'PolynomialKernel':
        detached_params = self._detach_params()
        return PolynomialKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                                degree=detached_params["degree"], opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        return polynomial_core_sparse(X1, X2, out,
                                      beta=dev_kernel_tensor_params["beta"],
                                      gamma=dev_kernel_tensor_params["gamma"],
                                      degree=dev_kernel_tensor_params["degree"], )

    def __str__(self):
        return f"PolynomialKernel(beta={self.beta}, gamma={self.gamma}, degree={self.degree})"

    def __repr__(self):
        return self.__str__()


class SigmoidKernel(DiffKernel):
    r"""Sigmoid (or hyperbolic tangent) kernel function, with additive and multiplicative constants.

    Follows the formula

    .. math::

        k(x, y) = \tanh(\alpha x^\top y + \beta)

    Parameters
    ----------
    beta : float-like
        Multiplicative constant
    gamma : float-like
        Multiplicative constant
    opt : Optional[FalkonOptions]
        Options which will be used in downstream kernel operations.


    """

    def __init__(self,
                 beta: Union[float, torch.Tensor],
                 gamma: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        self.beta = validate_diff_float(beta, param_name="beta")
        self.gamma = validate_diff_float(gamma, param_name="gamma")
        super().__init__("Sigmoid", opt, sigmoid_core, beta=self.beta, gamma=self.gamma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        return RuntimeError("SigmoidKernel is not implemented in KeOps")

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            warnings.warn("KeOps MMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            warnings.warn("KeOps dMMV implementation for %s kernel is not available. "
                          "Falling back to matrix-multiplication based implementation."
                          % (self.name))
        return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {}

    def detach(self) -> 'SigmoidKernel':
        detached_params = self._detach_params()
        return SigmoidKernel(beta=detached_params["beta"], gamma=detached_params["gamma"],
                             opt=self.params)

    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor,
                       **kwargs) -> torch.Tensor:
        dev_kernel_tensor_params = self._move_kernel_params(X1)
        return sigmoid_core_sparse(X1, X2, out,
                                   beta=dev_kernel_tensor_params["beta"],
                                   gamma=dev_kernel_tensor_params["gamma"], )

    def __str__(self):
        return f"SigmoidKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()

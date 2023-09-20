import warnings
from typing import Dict, Optional, Union

import torch

from falkon import sparse
from falkon.kernels import KeopsKernelMixin
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
            raise ValueError(f"Parameter {param_name} must be a scalar.") from None
    else:
        try:
            return torch.tensor([float(num)], dtype=torch.float64)
        except TypeError:
            raise TypeError(f"Parameter {param_name} must be a scalar or a tensor.") from None


def _dot_kernel_extra_mem(is_differentiable: bool, is_sparse: bool):
    base = {
        "0": 0,
    }
    extra_nm = 0
    # Normalize Matern to Gaussian or Laplacian
    if is_differentiable:
        extra_nm += 1  # To allocate out buffer
    elif is_sparse:
        # CUDA spspmm is impossible to evaluate. There is the output dense (which we don't
        # count here), the output sparse (assumed to be the same size as the dense n*m),
        # the various work buffers (for safety assume them to also be n*m).
        extra_nm += 2
    base["nm"] = extra_nm
    return base


def _dot_prod_calc(mat1: torch.Tensor, mat2: torch.Tensor, out: Optional[torch.Tensor], diag: bool) -> torch.Tensor:
    if diag:
        N, D = mat1.shape
        if out is None:
            out = torch.bmm(mat1.view(N, 1, D), mat2.view(N, D, 1)).reshape(-1)
        else:
            out = torch.bmm(mat1.view(N, 1, D), mat2.view(N, D, 1), out=out.reshape(-1, 1, 1))
            out = out.reshape(-1)
    else:
        if out is None:
            out = torch.mm(mat1, mat2.T)
        else:
            out = torch.mm(mat1, mat2.T, out=out)
    return out


def _sparse_dot_prod_calc(
    mat1: SparseTensor, mat2: SparseTensor, out: Optional[torch.Tensor], diag: bool
) -> torch.Tensor:
    if diag:
        return sparse.bdot(mat1, mat2, out)
    else:
        return sparse.sparse_matmul(mat1, mat2, out)


def linear_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, beta, gamma):
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    out = _dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    return out


def linear_core_sparse(
    mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor, diag: bool, beta, gamma
) -> torch.Tensor:
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    out = _sparse_dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    return out


def polynomial_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, beta, gamma, degree):
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    degree = degree.to(device=mat1.device, dtype=mat1.dtype)
    out = _dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    out.pow_(degree)
    return out


def polynomial_core_sparse(
    mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor, diag: bool, beta, gamma, degree
) -> torch.Tensor:
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    degree = degree.to(device=mat1.device, dtype=mat1.dtype)
    out = _sparse_dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    out.pow_(degree)
    return out


def sigmoid_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, beta, gamma):
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    out = _dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    out.tanh_()
    return out


def sigmoid_core_sparse(
    mat1: SparseTensor, mat2: SparseTensor, out: torch.Tensor, diag: bool, beta, gamma
) -> torch.Tensor:
    # Move hyper-parameters
    beta = beta.to(device=mat1.device, dtype=mat1.dtype)
    gamma = gamma.to(device=mat1.device, dtype=mat1.dtype)
    out = _sparse_dot_prod_calc(mat1, mat2, out, diag)
    out.mul_(gamma)
    out.add_(beta)
    out.tanh_()
    return out


class LinearKernel(DiffKernel, KeopsKernelMixin):
    """Linear Kernel with optional scaling and translation parameters.

    The kernel implemented here is the covariance function in the original
    input space (i.e. `X @ X.T`) with optional parameters to translate
    and scale the kernel: `beta + gamma * X @ X.T`

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
    >>> k = LinearKernel(beta=0.0, gamma=2.0)
    >>> X = torch.randn(100, 3)  # 100 samples in 3 dimensions
    >>> kernel_matrix = k(X, X)
    >>> torch.testing.assert_close(kernel_matrix, X @ X.T * 2)
    """

    def __init__(
        self,
        beta: Union[float, torch.Tensor] = 0.0,
        gamma: Union[float, torch.Tensor] = 1.0,
        opt: Optional[FalkonOptions] = None,
    ):
        beta = validate_diff_float(beta, param_name="beta")
        gamma = validate_diff_float(gamma, param_name="gamma")
        super().__init__("Linear", opt, linear_core, beta=beta, gamma=gamma)

    def keops_mmv_impl(self, X1, X2, v, kernel, out, opt, kwargs_m1, kwargs_m2):
        formula = "(gamma * (X | Y) + beta) * v"
        aliases = [
            "X = Vi(%d)" % (X1.shape[1]),
            "Y = Vj(%d)" % (X2.shape[1]),
            "v = Vj(%d)" % (v.shape[1]),
            "gamma = Pm(1)",
            "beta = Pm(1)",
        ]
        other_vars = [
            self.gamma.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.beta.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
        ]
        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self, is_differentiable, is_sparse, dtype, density1=None, density2=None) -> Dict[str, float]:
        return _dot_kernel_extra_mem(is_differentiable, is_sparse)

    def detach(self) -> "LinearKernel":
        return LinearKernel(beta=self.beta.detach(), gamma=self.gamma.detach(), opt=self.params)

    def compute_sparse(
        self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool, **kwargs
    ) -> torch.Tensor:
        return linear_core_sparse(X1, X2, out, diag, beta=self.beta, gamma=self.gamma)

    def __str__(self):
        return f"LinearKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()


class PolynomialKernel(DiffKernel, KeopsKernelMixin):
    r"""Polynomial kernel with multiplicative and additive constants.

    Follows the formula

    .. math::

        (\gamma * X_1^\top X_2 + \beta)^{\mathrm{degree}}

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

    def __init__(
        self,
        beta: Union[float, torch.Tensor],
        gamma: Union[float, torch.Tensor],
        degree: Union[float, torch.Tensor],
        opt: Optional[FalkonOptions] = None,
    ):
        beta = validate_diff_float(beta, param_name="beta")
        gamma = validate_diff_float(gamma, param_name="gamma")
        degree = validate_diff_float(degree, param_name="degree")
        super().__init__("Polynomial", opt, polynomial_core, beta=beta, gamma=gamma, degree=degree)

    def keops_mmv_impl(self, X1, X2, v, kernel, out, opt, kwargs_m1, kwargs_m2):
        formula = "Powf((gamma * (X | Y) + beta), degree) * v"
        aliases = [
            "X = Vi(%d)" % (X1.shape[1]),
            "Y = Vj(%d)" % (X2.shape[1]),
            "v = Vj(%d)" % (v.shape[1]),
            "gamma = Pm(1)",
            "beta = Pm(1)",
            "degree = Pm(1)",
        ]
        other_vars = [
            self.gamma.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.beta.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
            self.degree.reshape(1, 1).to(dtype=X1.dtype, device=X1.device),
        ]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self, is_differentiable, is_sparse, dtype, density1=None, density2=None) -> Dict[str, float]:
        return _dot_kernel_extra_mem(is_differentiable, is_sparse)

    def detach(self) -> "PolynomialKernel":
        return PolynomialKernel(
            beta=self.beta.detach(), gamma=self.gamma.detach(), degree=self.degree.detach(), opt=self.params
        )

    def compute_sparse(
        self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool, **kwargs
    ) -> torch.Tensor:
        return polynomial_core_sparse(X1, X2, out, diag, beta=self.beta, gamma=self.gamma, degree=self.degree)

    def __str__(self):
        return f"PolynomialKernel(beta={self.beta}, gamma={self.gamma}, degree={self.degree})"

    def __repr__(self):
        return self.__str__()


class SigmoidKernel(DiffKernel, KeopsKernelMixin):
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

    def __init__(
        self,
        beta: Union[float, torch.Tensor],
        gamma: Union[float, torch.Tensor],
        opt: Optional[FalkonOptions] = None,
    ):
        beta = validate_diff_float(beta, param_name="beta")
        gamma = validate_diff_float(gamma, param_name="gamma")
        super().__init__("Sigmoid", opt, sigmoid_core, beta=beta, gamma=gamma)

    def keops_mmv_impl(self, X1, X2, v, kernel, out, opt, kwargs_m1, kwargs_m2):
        return RuntimeError("SigmoidKernel is not implemented in KeOps")

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            warnings.warn(
                "KeOps MMV implementation for %s kernel is not available. "
                "Falling back to matrix-multiplication based implementation." % self.name
            )
        return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            warnings.warn(
                "KeOps dMMV implementation for %s kernel is not available. "
                "Falling back to matrix-multiplication based implementation." % self.name
            )
        return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def extra_mem(self, is_differentiable, is_sparse, dtype, density1=None, density2=None) -> Dict[str, float]:
        return _dot_kernel_extra_mem(is_differentiable, is_sparse)

    def detach(self) -> "SigmoidKernel":
        return SigmoidKernel(beta=self.beta, gamma=self.gamma, opt=self.params)

    def compute_sparse(
        self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool, **kwargs
    ) -> torch.Tensor:
        return sigmoid_core_sparse(X1, X2, out, diag, beta=self.beta, gamma=self.gamma)

    def __str__(self):
        return f"SigmoidKernel(beta={self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return self.__str__()

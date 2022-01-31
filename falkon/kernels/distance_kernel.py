from typing import Union, Optional, Dict

import torch

from falkon import sparse
from falkon.kernels.diff_kernel import DiffKernel
from falkon.la_helpers.square_norm_fn import square_norm_diff
from falkon.options import FalkonOptions
from falkon.sparse import SparseTensor

SQRT3 = 1.7320508075688772
SQRT5 = 2.23606797749979


def validate_sigma(sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        # Sigma is a 1-item tensor ('single')
        try:
            sigma.item()
            return sigma
        except ValueError:
            pass
        # Sigma is a vector ('diag')
        if sigma.dim() == 1 or sigma.shape[1] == 1:
            return sigma.reshape(-1)
        else:
            # TODO: Better error
            raise ValueError("sigma must be a scalar or a vector.")
    else:
        try:
            print("Converting sigma to float64")
            return torch.tensor([float(sigma)], dtype=torch.float64)
        except TypeError:
            raise TypeError("Sigma must be a scalar or a tensor.")


def _sq_dist(mat1, mat2, norm_mat1, norm_mat2, out: Optional[torch.Tensor]) -> torch.Tensor:
    if mat1.dim() == 3:
        if out is None:
            out = torch.baddbmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1)  # b*n*m
        else:
            out = torch.baddbmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1,
                                out=out)  # b*n*m
    else:
        if out is None:
            out = torch.addmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1)  # n*m
        else:
            out = torch.addmm(norm_mat1, mat1, mat2.transpose(-2, -1), alpha=-2, beta=1,
                              out=out)  # n*m
    out.add_(norm_mat2.transpose(-2, -1))
    out.clamp_min_(1e-20)
    return out


def _sparse_sq_dist(X1_csr: SparseTensor, X2_csr: SparseTensor,
                    X1: SparseTensor, X2: SparseTensor,
                    out: torch.Tensor) -> torch.Tensor:
    sq1 = torch.empty(X1_csr.size(0), dtype=X1_csr.dtype, device=X1_csr.device)
    sparse.sparse_square_norm(X1_csr, sq1)  # TODO: This must be implemented for CUDA tensors
    sq1 = sq1.reshape(-1, 1)
    sq2 = torch.empty(X2_csr.size(0), dtype=X2_csr.dtype, device=X2_csr.device)
    sparse.sparse_square_norm(X2_csr, sq2)
    sq2 = sq2.reshape(-1, 1)
    sparse.sparse_matmul(X1, X2, out)
    out.mul_(-2.0)
    out.add_(sq1.to(device=X1.device))
    out.add_(sq2.to(device=X2.device).t())
    out.clamp_min_(1e-20)
    return out


def _distancek_diag(mat1, out: Optional[torch.Tensor]):
    if out is None:
        return torch.ones(mat1.shape[0], device=mat1.device, dtype=mat1.dtype)

    out.fill_(1.0)
    return out


def rbf_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, sigma):
    """
    Note 1: if out is None, then this function will be differentiable wrt all three remaining inputs.
    Note 2: this function can deal with batched inputs

    Parameters
    ----------
    sigma
    mat1
    mat2
    out

    Returns
    -------

    """
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1 or n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1 or m*1

    out = _sq_dist(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    out.mul_(-0.5)
    out.exp_()
    return out


def rbf_core_sparse(mat1: SparseTensor, mat2: SparseTensor,
                    mat1_csr: SparseTensor, mat2_csr: SparseTensor,
                    out: torch.Tensor, diag: bool, sigma) -> torch.Tensor:
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    gamma = 0.5 / (sigma ** 2)
    out = _sparse_sq_dist(X1_csr=mat1_csr, X2_csr=mat2_csr, X1=mat1, X2=mat2, out=out)
    out.mul_(-gamma)
    out.exp_()
    return out


def laplacian_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, sigma):
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1
    orig_out = out
    out = _sq_dist(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    out.sqrt_()  # Laplacian: sqrt of squared-difference
    # The gradient calculation needs the output of sqrt_
    if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
        out = out.neg()
    else:
        out.neg_()
    out.exp_()
    return out


def laplacian_core_sparse(mat1: SparseTensor, mat2: SparseTensor,
                          mat1_csr: SparseTensor, mat2_csr: SparseTensor,
                          out: torch.Tensor, diag: bool, sigma) -> torch.Tensor:
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    gamma = 1 / sigma
    out = _sparse_sq_dist(X1_csr=mat1_csr, X2_csr=mat2_csr, X1=mat1, X2=mat2, out=out)
    out.sqrt_()
    out.mul_(-gamma)
    out.exp_()
    return out


def matern_core(mat1, mat2, out: Optional[torch.Tensor], diag: bool, sigma, nu):
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    if nu == 0.5:
        return laplacian_core(mat1, mat2, out, diag, sigma)
    elif nu == float('inf'):
        return rbf_core(mat1, mat2, out, diag, sigma)
    orig_out = out
    mat1_div_sig = mat1 / sigma
    mat2_div_sig = mat2 / sigma
    norm_sq_mat1 = square_norm_diff(mat1_div_sig, -1, True)  # b*n*1
    norm_sq_mat2 = square_norm_diff(mat2_div_sig, -1, True)  # b*m*1

    out = _sq_dist(mat1_div_sig, mat2_div_sig, norm_sq_mat1, norm_sq_mat2, out)
    if nu == 1.5:
        # (1 + sqrt(3)*D) * exp(-sqrt(3)*D))
        out.sqrt_()
        if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
            out = out.mul(SQRT3)
        else:
            out.mul_(SQRT3)
        out_neg = torch.neg(out)  # extra n*m block
        out_neg.exp_()
        out.add_(1.0).mul_(out_neg)
    elif nu == 2.5:
        # (1 + sqrt(5)*D + (sqrt(5)*D)^2 / 3 ) * exp(-sqrt(5)*D)
        out_sqrt = torch.sqrt(out)
        if orig_out is None:  # TODO: We could be more explicit in the parameters about whether the gradient is or isn't needed
            out_sqrt = out_sqrt.mul(SQRT5)
        else:
            out_sqrt.mul_(SQRT5)
        out.mul_(5.0 / 3.0).add_(out_sqrt).add_(1.0)
        out_sqrt.neg_().exp_()
        out.mul_(out_sqrt)

    return out


def matern_core_sparse(mat1: SparseTensor, mat2: SparseTensor,
                       mat1_csr: SparseTensor, mat2_csr: SparseTensor,
                       out: torch.Tensor, diag: bool, sigma, nu) -> torch.Tensor:
    if diag:
        return _distancek_diag(mat1, out)
    # Move hparams
    sigma = sigma.to(device=mat1.device, dtype=mat1.dtype)
    if nu == 0.5:
        return laplacian_core_sparse(mat1, mat2, mat1_csr, mat2_csr, out, diag, sigma)
    elif nu == float('inf'):
        return rbf_core_sparse(mat1, mat2, mat1_csr, mat2_csr, out, diag, sigma)
    gamma = 1 / (sigma ** 2)
    out = _sparse_sq_dist(X1_csr=mat1_csr, X2_csr=mat2_csr, X1=mat1, X2=mat2, out=out)
    out.mul_(gamma)

    # For certain nu = 1.5, 2.5 we will need an extra n*m block
    if nu == 1.5:
        # (1 + sqrt(3)*D) * exp(-sqrt(3)*D))
        out.sqrt_()
        out.mul_(SQRT3)
        out_neg = torch.neg(out)
        out_neg.exp_()
        out.add_(1.0).mul_(out_neg)
    elif nu == 2.5:
        # (1 + sqrt(5)*D + (sqrt(5)*D)^2 / 3 ) * exp(-sqrt(5)*D)
        out_sqrt = torch.sqrt(out)
        out_sqrt.mul_(SQRT5)
        out.mul_(5.0 / 3.0).add_(out_sqrt).add_(1.0)
        out_sqrt.neg_().exp_()
        out.mul_(out_sqrt)
    return out


class GaussianKernel(DiffKernel):
    r"""Class for computing the Gaussian kernel and related kernel-vector products

    The Gaussian kernel is one of the most common and effective kernel embeddings
    since it is infinite dimensional, and governed by a single parameter. The kernel length-scale
    determines the width of the Gaussian distribution which is placed on top of each point.
    A larger sigma corresponds to a wide Gaussian, so that the relative influence of far away
    points will be high for computing the kernel at a given datum.
    On the opposite side of the spectrum, a small sigma means that only nearby points will
    influence the kernel.

    Parameters
    -----------
    sigma
        The length-scale of the kernel.
        This can be a scalar, and then it corresponds to the standard deviation
        of the Gaussian distribution from which the kernel is derived.
        If `sigma` is a vector of size `d` (where `d` is the dimensionality of the data), it is
        interpreted as the diagonal standard deviation of the Gaussian distribution.
        It can also be a matrix of  size `d*d` where `d`, in which case sigma will be the precision
        matrix (inverse covariance).
    opt
        Additional options to be forwarded to the matrix-vector multiplication
        routines.

    Examples
    --------
    Creating a Gaussian kernel with a single length-scale. Operations on this kernel will not
    use KeOps.

    >>> K = GaussianKernel(sigma=3.0, opt=FalkonOptions(keops_active="no"))

    Creating a Gaussian kernel with a different length-scale per dimension

    >>> K = GaussianKernel(sigma=torch.tensor([1.0, 3.5, 7.0]))

    Creating a Gaussian kernel object with full covariance matrix (randomly chosen)

    >>> mat = torch.randn(3, 3, dtype=torch.float64)
    >>> sym_mat = mat @ mat.T
    >>> K = GaussianKernel(sigma=sym_mat)
    >>> K
    GaussianKernel(sigma=tensor([[ 2.0909,  0.0253, -0.2490],
            [ 0.0253,  0.3399, -0.5158],
            [-0.2490, -0.5158,  4.4922]], dtype=torch.float64))  #random


    Notes
    -----
    The Gaussian kernel with a single length-scale follows

    .. math::

        k(x, x') = \exp{-\dfrac{\lVert x - x' \rVert^2}{2\sigma^2}}


    When the length-scales are specified as a matrix, the RBF kernel is determined by

    .. math::

        k(x, x') = \exp{-\dfrac{1}{2}x\Sigma x'}


    In both cases, the actual computation follows a different path, working on the expanded
    norm.
    """
    kernel_name = "gaussian"
    core_fn = rbf_core

    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        sigma = validate_sigma(sigma)
        super().__init__(self.kernel_name, opt, core_fn=GaussianKernel.core_fn, sigma=sigma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(SqDist(x1 / g, x2 / g) * IntInv(-2)) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {
            # Data-matrix / sigma in prepare + Data-matrix / sigma in apply
            'nd': 2,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }

    def detach(self) -> 'GaussianKernel':
        return GaussianKernel(self.sigma.detach(), opt=self.params)

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        return rbf_core_sparse(X1, X2, X1_csr, X2_csr, out, diag, self.sigma)

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Gaussian kernel<{self.sigma}>"


class LaplacianKernel(DiffKernel):
    r"""Class for computing the Laplacian kernel, and related kernel-vector products.

    The Laplacian kernel is similar to the Gaussian kernel, but less sensitive to changes
    in the parameter `sigma`.

    Parameters
    ----------
    sigma
        The length-scale of the Laplacian kernel

    Notes
    -----
    The Laplacian kernel is determined by the following formula

    .. math::

        k(x, x') = \exp{-\frac{\lVert x - x' \rVert}{\sigma}}

    """
    kernel_name = "laplacian"

    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        sigma = validate_sigma(sigma)

        super().__init__(self.kernel_name, opt, core_fn=laplacian_core, sigma=sigma)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(-Sqrt(SqDist(x1 / g, x2 / g))) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        return {
            # Data-matrix / sigma in prepare + Data-matrix / sigma in apply
            'nd': 2,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }

    def detach(self) -> 'LaplacianKernel':
        return LaplacianKernel(self.sigma.detach(), opt=self.params)

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        return laplacian_core_sparse(X1, X2, X1_csr, X2_csr, out, diag, self.sigma)

    def __repr__(self):
        return f"LaplacianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"Laplaciankernel<{self.sigma}>"


class MaternKernel(DiffKernel):
    r"""Class for computing the Matern kernel, and related kernel-vector products.

    The Matern kernels define a generic class of kernel functions which includes the
    Laplacian and Gaussian kernels. The class is parametrized by 'nu'. When `nu = 0.5`
    this kernel is equivalent to the Laplacian kernel, when `nu = float('inf')`, the
    Matern kernel is equivalent to the Gaussian kernel.

    This class implements the Matern kernel only for the values of nu which have a closed
    form solution, which are 0.5, 1.5, 2.5, and infinity.

    Parameters
    ----------
    sigma
        The length-scale of the Matern kernel. The length-scale can be either a scalar
        or a vector. Matrix-valued length-scales are not allowed for the Matern kernel.
    nu
        The parameter of the Matern kernel. It should be one of `0.5`, `1.5`, `2.5` or
        `inf`.

    Notes
    -----
    While for `nu = float('inf')` this kernel is equivalent to the :class:`GaussianKernel`,
    the implementation is more general and using the :class:`GaussianKernel` directly
    may be computationally more efficient.

    """
    _valid_nu_values = frozenset({0.5, 1.5, 2.5, float('inf')})

    def __init__(self,
                 sigma: Union[float, torch.Tensor],
                 nu: Union[float, torch.Tensor],
                 opt: Optional[FalkonOptions] = None):
        sigma = validate_sigma(sigma)
        nu = self.validate_nu(nu)
        self.kernel_name = f"{nu:.1f}-matern"
        super().__init__(self.kernel_name, opt, core_fn=matern_core, sigma=sigma, nu=nu)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        if self.nu == 0.5:
            formula = 'Exp(-Norm2(x1 / s - x2 / s)) * v'
        elif self.nu == 1.5:
            formula = ('(IntCst(1) + Sqrt(IntCst(3)) * Norm2(x1 / s - x2 / s)) * '
                       '(Exp(-Sqrt(IntCst(3)) * Norm2(x1 / s - x2 / s)) * v)')
        elif self.nu == 2.5:
            formula = ('(IntCst(1) + Sqrt(IntCst(5)) * Norm2(x1 / s - x2 / s) + '
                       '(IntInv(3) * IntCst(5)) * SqNorm2(x1 / s - x2 / s)) * '
                       '(Exp(-Sqrt(IntCst(5)) * Norm2(x1 / s - x2 / s)) * v)')
        elif self.nu == float('inf'):
            formula = 'Exp(IntInv(-2) * SqDist(x1 / s, x2 / s)) * v'
        else:
            raise RuntimeError(f"Unrecognized value of nu ({self.nu}). "
                               f"The onnly allowed values are 0.5, 1.5, 2.5, inf.")
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            's = Pm(%d)' % (self.sigma.shape[0])
        ]
        other_vars = [self.sigma.to(device=X1.device, dtype=X1.dtype)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def extra_mem(self) -> Dict[str, float]:
        extra_mem = {
            # Data-matrix / sigma
            'nd': 1,
            'md': 1,
            # Norm results in prepare
            'm': 1,
            'n': 1,
        }
        if self.nu in {1.5, 2.5}:
            # Extra kernel block in transform
            extra_mem['nm'] = 1
        return extra_mem

    def detach(self) -> 'MaternKernel':
        return MaternKernel(self.sigma.detach(), self.nondiff_params["nu"], opt=self.params)

    @staticmethod
    def validate_nu(nu: Union[torch.Tensor, float]) -> float:
        if isinstance(nu, torch.Tensor):
            if nu.requires_grad:
                raise ValueError("The nu parameter of the Matern kernel is not differentiable, "
                                 "and must not require gradients.")
            try:
                out_nu = round(nu.item(), ndigits=2)
            except ValueError:
                raise ValueError("nu=%s is not convertible to a scalar." % (nu))
        elif isinstance(nu, float):
            out_nu = round(nu, ndigits=2)
        else:
            raise TypeError(f"nu must be a float or a tensor, not a {type(nu)}")
        if out_nu not in MaternKernel._valid_nu_values:
            raise ValueError(f"The given value of nu = {out_nu} can only take "
                             f"values {MaternKernel._valid_nu_values}.")
        return out_nu

    # noinspection PyMethodOverriding
    def compute_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor, diag: bool,
                       X1_csr: SparseTensor, X2_csr: SparseTensor) -> torch.Tensor:
        if len(self.sigma) > 1:
            raise NotImplementedError("Sparse kernel is only implemented for scalar sigmas.")
        return matern_core_sparse(X1, X2, X1_csr, X2_csr, out, diag,
                                  self.sigma, self.nondiff_params['nu'])

    def __repr__(self):
        return f"MaternKernel(sigma={self.sigma}, nu={self.nondiff_params['nu']:.1f})"

    def __str__(self):
        return f"Matern kernel<{self.sigma}, {self.nondiff_params['nu']:.1f}>"

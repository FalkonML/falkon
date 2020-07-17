import dataclasses
from abc import abstractmethod, ABC
from typing import Optional

import torch

from falkon.options import FalkonOptions
import falkon

__all__ = ("Loss", "LogisticLoss")


class Loss(ABC):
    """Abstract generalized self-concordant loss function class.

    Such loss functions must be three times differentiable; but for the LogFalkon algorithm
    only the first two derivatives are used.
    Subclasses must implement the `__call__` method
    which calculates the loss function given two input vectors (the inputs could also be
    matrices e.g. for the softmax loss), the `df` method which calculates the first derivative
    of the function and `ddf` which calculates the second derivative.

    Additionally, this class provides two methods (`knmp_grad` and `knmp_hess`) which calculate
    kernel-vector products using the loss derivatives for vectors. These functions are specific
    to the LogFalkon algorithm.

    Parameters
    -----------
    name
        A descriptive name for the loss function (e.g. "logistic", "softmax")
    kernel
        The kernel function used for training a LogFalkon model
    opt
        Falkon options container. Will be passed to the kernel when computing kernel-vector
        products.

    See Also
    --------
    :class:`LogisticLoss` : a concrete implementation of this class for the logistic loss.
    """

    def __init__(self,
                 name: str,
                 kernel: falkon.kernels.Kernel,
                 opt: FalkonOptions = FalkonOptions()):
        self.name = name
        self.kernel = kernel
        self.params = opt

    def _update_opt(self, opt: Optional[FalkonOptions]):
        new_opt = self.params
        if opt is not None:
            new_opt = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        return new_opt

    @abstractmethod
    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    def knmp_grad(self, X, Xc, Y, u, opt=None):
        """
        Calculate (1/n)* K(Xc, X) @ df(Y, K(X, Xc) @ u)

        And return also K(X, Xc) @ u
        """
        opt = self._update_opt(opt)
        func_val = self.kernel.mmv(X, Xc, u, opt=opt)
        grad = self.df(Y, func_val)
        out = self.kernel.mmv(Xc, X, grad, opt=opt)
        out.mul_(1 / X.shape[0])
        return out, func_val

    def knmp_hess(self, X, Xc, Y, f, u, opt=None):
        r"""Compute a kernel-vector product with a rescaling with the second derivative

        Given kernel function :math:`K`, the loss represented by this class :math:`\mathcal{l}`,
        number of samples :math:`n`, this function follows equation

        .. math::

        \dfrac{1}{n} K(X_c, X) @ (\mathcal{l}^{''} * K(X, X_c) @ u)


        Parameters
        ----------
        X : Tensor (N x D)
            Data matrix
        Xc : Tensor (M x D)
            Matrix of Nystroem centers
        Y : Tensor (N x 1)
            The targets with respect to which the loss is computed
        f : Tensor (N x 1)
            Current predictions for the function, for which the loss is computed
        u : Tensor (M x 1)
            Vector
        """
        opt = self._update_opt(opt)
        inner = self.kernel.mmv(X, Xc, u, opt=opt)
        inner.mul_(self.ddf(Y, f))
        outer = self.kernel.mmv(Xc, X, inner, opt=opt)
        outer.mul_(1 / X.shape[0])
        return outer

    def __str__(self):
        return self.name


class LogisticLoss(Loss):
    """Wrapper for the logistic loss, to be used in conjunction with the `LogisticFalkon` estimator.

    Parameters
    -----------
    kernel
        The kernel function used for training a :class:`LogisticFalkon` model
    opt
        Falkon options container. Will be passed to the kernel when computing kernel-vector
        products.

    Examples
    --------

    >>> k = falkon.kernels.GaussianKernel(3)
    >>> log_loss = LogisticLoss(k)
    >>> estimator = falkon.LogisticFalkon(k, [1e-4, 1e-4, 1e-4], [3, 3, 3], loss=log_loss, M=100)

    """

    def __init__(self, kernel: falkon.kernels.Kernel, opt: FalkonOptions = FalkonOptions()):
        super().__init__(name="LogisticLoss", kernel=kernel, opt=opt)

    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Compute the logistic loss between two 1-dimensional tensors

        The formula used is :math:`\\log(1 + \\exp(-y_1 * y_2))`

        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        loss
            The logistic loss between the two input vectors.
        """
        return torch.log(1 + torch.exp(-y1 * y2))

    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the logistic loss between two vectors

        The formula used is

        ..math::

            -y_1 / (1 + \\exp{y_1 * y_2})

        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        d_loss
            The derivative of the logistic loss, calculated between the two input vectors.
        """
        out = -y1
        div = y1 * y2
        div.exp_().add_(1)
        out.div_(div)
        return out

    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Compute the second derivative of the logistic loss between two vectors

        The formula used is

        ..math::

            y_1^2 \\dfrac{1}{1 + \\exp{-y_1 * y_2}} \\dfrac{1}{1 + \\exp{y_1 * y_2}}


        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        dd_loss
            The second derivative of the logistic loss, calculated between the two input vectors.
        """
        out = torch.pow(y1, 2)
        mul = y1 * y2
        mul.exp_()

        div = mul.reciprocal()
        div.add_(mul).add_(2)
        out.div_(div)
        return out

    def __repr__(self):
        return "LogisticLoss(kernel=%r)" % (self.kernel)
